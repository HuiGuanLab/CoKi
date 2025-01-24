import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from .resnet import get_model

lora_config = LoraConfig(
    r=16,
    target_modules=['k_proj', 'v_proj', 'q_proj', 'out_proj', 'fc1', 'fc2'],
    lora_alpha=32,
    lora_dropout=0,
    bias='none',
)


class CoKi(nn.Module):
    def __init__(self, cfg):
        super(CoKi, self).__init__()
        self.choices = nn.ModuleDict({
            'global': nn.ModuleDict({
                'attrnet': AttrEmbedding(cfg.DATA.NUM_ATTRIBUTES, cfg.MODEL.ATTRIBUTE.EMBED_SIZE),
                # 'basenet': get_model(cfg.MODEL.GLOBAL.BACKBONE.NAME, pretrained=True),
                'basenet': get_peft_model(CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16"), lora_config),
                'textnet': CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16"),
                'mmattnet': MultiModalAttention(
                    embed_dim=cfg.MODEL.EMBED_SIZE,
                    num_heads=cfg.MODEL.MULTIHEAD.NUM_HEADS,
                    attr_embed_size=cfg.MODEL.ATTRIBUTE.EMBED_SIZE
                )
            })})

        for param in self.choices['global']['textnet'].parameters():
            param.requires_grad = False

        if cfg.MODEL.LOCAL.ENABLE:
            self.choices.update({
                'local': nn.ModuleDict({
                    'attrnet': self.choices['global']['attrnet'],
                    'basenet': get_model(cfg.MODEL.LOCAL.BACKBONE.NAME, pretrained=True),
                    'attnnet': AttnEmbedding(
                        cfg.MODEL.ATTRIBUTE.EMBED_SIZE,
                        cfg.MODEL.LOCAL.BACKBONE.EMBED_SIZE,
                        cfg.MODEL.LOCAL.ATTENTION.SPATIAL.COMMON_EMBED_SIZE,
                        cfg.MODEL.LOCAL.ATTENTION.CHANNEL.REDUCTION_RATE,
                        cfg.MODEL.EMBED_SIZE,
                        cfg.MODEL.LOCAL.ATTENTION.SPATIAL.ENABLE,
                        cfg.MODEL.LOCAL.ATTENTION.CHANNEL.ENABLE
                    )
                })})

    def forward(self, x, a, t, level='global', mask_value=None, zeroshot_mode=False):
        # print("x shape: ", x.shape)  # (B, 3, 224, 224)
        # print("a shape: ", a.shape)  # (B)
        # print("t shape: ", t.shape)  # (B, L_t)
        x = self.choices[level]['basenet'](x).last_hidden_state[:, 1:, :]
        x = self.choices[level]['basenet'].visual_projection(x)
        # print("x shape: ", x.shape)  # (B, L_x, D)

        if a is not None:
            a_embed = self.choices[level]['attrnet'](a)
            if zeroshot_mode:
                mask = ~torch.isin(a, torch.tensor(mask_value, device=a.device))
                mask = mask.float().unsqueeze(1)
                a = a_embed * mask
            else:
                a = a_embed
        # print("a shape: ", a.shape)  # (B, D)

        if t is not None:
            with torch.no_grad():
                eos_idx = t.argmax(dim=-1)
                t = self.choices[level]['textnet'](t).last_hidden_state[torch.arange(t.size(0)), eos_idx]
                t = self.choices[level]['textnet'].text_projection(t)
        # print("t shape: ", t.shape)  # (B, D)

        mmfeat, attn_weights = self.choices[level]['mmattnet'](x, a, t, zeroshot_mode=zeroshot_mode)
        # print("mmfeat shape: ", mmfeat.shape)  # (B, D)
        # print("attn_weights shape: ", attn_weights.shape)  # (B, 2, L_x)

        return mmfeat, attn_weights

    def get_attr_embedding(self, a):
        if a is not None:
            a_embed = self.choices['global']['attrnet'](a)
            a_embed = self.choices['global']['mmattnet'].attr_linear(a_embed)
        else:
            a_embed = None
        return a_embed

    def get_text_embedding(self, t):
        if t is not None:
            eos_idx = t.argmax(dim=-1)
            t_embed = self.choices['global']['textnet'](t).last_hidden_state[torch.arange(t.size(0)), eos_idx]
            t_embed = self.choices['global']['textnet'].text_projection(t_embed)
            t_embed = self.choices['global']['mmattnet'].text_linear(t_embed)
        else:
            t_embed = None
        return t_embed

    def load_state_dict(self, loaded_state_dict):
        state = super(CoKi, self).state_dict()
        for k in loaded_state_dict:
            if k in state:
                state[k] = loaded_state_dict[k]
        super(CoKi, self).load_state_dict(state)


class AttrEmbedding(nn.Module):
    def __init__(self, n_attrs, embed_size):
        super(AttrEmbedding, self).__init__()
        self.attr_embedding = torch.nn.Embedding(n_attrs, embed_size)

    def forward(self, x):
        return self.attr_embedding(x)


class AttnEmbedding(nn.Module):
    def __init__(
            self,
            attr_embed_size,
            img_embed_size,
            common_embed_size,
            reduction_rate,
            embed_size,
            spatial_en=True,
            channel_en=True):
        super(AttnEmbedding, self).__init__()

        self.spatial_en = spatial_en
        self.channel_en = channel_en

        if self.spatial_en:
            self.attr_transform1 = nn.Linear(
                attr_embed_size,
                common_embed_size
            )
            self.conv = nn.Conv2d(
                img_embed_size,
                common_embed_size,
                kernel_size=1, stride=1
            )

        if self.channel_en:
            self.attr_transform2 = nn.Linear(
                attr_embed_size,
                attr_embed_size
            )
            self.fc1 = nn.Linear(
                img_embed_size+attr_embed_size,
                img_embed_size//reduction_rate
            )
            self.fc2 = nn.Linear(
                img_embed_size//reduction_rate,
                img_embed_size
            )

        self.feature_fc = nn.Linear(
            img_embed_size,
            embed_size
        )

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.aapool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, a):
        if self.spatial_en:
            attmap = self.spatial_attn(x, a)

            x = x * attmap
            x = x.view(x.size(0), x.size(1), -1)
            x = x.sum(dim=2)
        else:
            x = self.aapool(x).squeeze()

        if self.channel_en:
            m = self.channel_attn(x, a)
            x = x * m

        x = self.feature_fc(x)

        return x, attmap.squeeze() if self.spatial_en else None

    def spatial_attn(self, x, a):
        x = self.conv(x)
        x = self.tanh(x)

        a = self.attr_transform1(a)
        a = self.tanh(a)
        a = a.view(a.size(0), a.size(1), 1, 1)
        a = a.expand_as(x)

        attmap = a * x
        attmap = torch.sum(attmap, dim=1, keepdim=True)
        attmap = torch.div(attmap, x.size(1) ** 0.5)
        attmap = attmap.view(attmap.size(0), attmap.size(1), -1)
        attmap = self.softmax(attmap)
        attmap = attmap.view(attmap.size(0), attmap.size(1), x.size(2), x.size(3))

        return attmap

    def channel_attn(self, x, a):
        a = self.attr_transform2(a)
        a = self.relu(a)

        cnt = torch.cat((x, a), dim=1)
        m = self.fc1(cnt)
        m = self.relu(m)
        m = self.fc2(m)
        m = self.sigmoid(m)

        return m


class MultiModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attr_embed_size):
        super(MultiModalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_modality_prompt = True
        self.use_modality_imputer = True

        self.attr_linear = nn.Linear(attr_embed_size, embed_dim)
        self.text_linear = nn.Linear(embed_dim, embed_dim)
        self.query_linear = nn.Linear(3 * embed_dim, embed_dim)

        self.modality_prompt = nn.Embedding(3, embed_dim)
        self.modality_imputer = nn.Embedding(2, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x, a=None, t=None, zeroshot_mode=False):
        batch_size = x.size(0)

        if self.use_modality_prompt:
            prompt_ids = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            if a is not None and t is None:
                prompt_ids[:] = 1
            if t is not None and a is None:
                prompt_ids[:] = 2
            prompt_token = self.modality_prompt(prompt_ids)
        else:
            prompt_token = torch.zeros((batch_size, self.embed_dim), device=x.device)

        if zeroshot_mode:
            # zero-shot 时，根据是否全 0 来决定用 imputer 还是真实特征
            zs_mask = (a == 0).all(dim=1)

            if a is not None:
                a_linear = self.attr_linear(a)  # (B, D)
                missing_a = self.modality_imputer(
                    torch.zeros(1, dtype=torch.long, device=x.device)
                ).expand(batch_size, -1)  # (B, D)
                
                # (B, D) -> 根据 zs_mask 替换
                a_processed = torch.where(
                    zs_mask.unsqueeze(1),  # (B, 1)
                    missing_a,             # 被屏蔽的样本使用 imputer
                    a_linear               # 正常样本使用属性特征
                )
            else:
                # 如果 a=None，又想使用 imputer，则直接全部 imputer
                a_processed = self.modality_imputer(
                    torch.zeros(batch_size, dtype=torch.long, device=x.device)
                )  # (B, D)
            
            if t is not None:
                t_linear = self.text_linear(t)
                missing_t = self.modality_imputer(
                    torch.ones(1, dtype=torch.long, device=x.device)
                ).expand(batch_size, -1)

                t_processed = torch.where(
                    zs_mask.unsqueeze(1),
                    missing_t,
                    t_linear
                )
            else:
                t_processed = self.modality_imputer(
                    torch.ones(batch_size, dtype=torch.long, device=x.device)
                )

        else:
            # 非 zero-shot 模式下，直接用输入特征或 imputer/零向量替代
            if a is not None:
                a_processed = self.attr_linear(a)
            else:
                if self.use_modality_imputer:
                    a_processed = self.modality_imputer(
                        torch.zeros(batch_size, dtype=torch.long, device=x.device)
                    )
                else:
                    a_processed = torch.zeros((batch_size, self.embed_dim), device=x.device)

            if t is not None:
                t_processed = self.text_linear(t)
            else:
                if self.use_modality_imputer:
                    t_processed = self.modality_imputer(
                        torch.ones(batch_size, dtype=torch.long, device=x.device)
                    )
                else:
                    t_processed = torch.zeros((batch_size, self.embed_dim), device=x.device)


        q_cat = torch.cat([prompt_token, a_processed, t_processed], dim=1)  # (B, 3D)
        q = self.query_linear(q_cat).unsqueeze(1)  # (B, 1, D)

        kv = x  # (B, L_x, D)

        attn_output, attn_weights = self.multihead_attn(query=q, key=kv, value=kv)
        attn_output = self.layer_norm(attn_output)

        attn_output = torch.mean(attn_output, dim=1)

        return attn_output, attn_weights
