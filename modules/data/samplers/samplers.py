import math
import os
import random

from torch.utils.data.sampler import Sampler


class TripletSampler(Sampler):
    def __init__(self, cfg):
        self.num_triplets = cfg.DATA.NUM_TRIPLETS
        self.batch_size = cfg.DATA.TRAIN_BATCHSIZE
        self.attrs = cfg.DATA.ATTRIBUTES.NAME
        self.num_values = cfg.DATA.ATTRIBUTES.NUM
        self.attr_desc = {
            "skirt_length": "the skirt's length, affecting the level of leg visibility and overall silhouette, ranging from shorter to longer styles",
            "sleeve_length": "the sleeve length, determining arm coverage and influencing the outfit's overall balance between exposure and coverage",
            "coat_length": "the coat's length, shaping the upper body coverage and overall appearance, with styles that range from shorter cuts to longer, fuller lengths",
            "pant_length": "the length of the pants, impacting leg coverage and silhouette, with variations from shorter cuts to full-length options",
            "collar_design": "the collar's design, which frames the neckline in different ways, offering both traditional and contemporary looks",
            "lapel_design": "the lapel style, shaping the coat's opening and contributing to either a structured or flowing look around the neck and shoulders",
            "neckline_design": "the neckline shape, defining how the neck and shoulders are highlighted, with options from minimalist to more pronounced designs",
            "neck_design": "the neck area's style, affecting how the neck is framed and the balance between openness and coverage around the neck",

            "clothes category": "the general category of the garment, identifying its primary purpose or function, such as outerwear, tops, or bottoms",
            "clothes button": "the presence and arrangement of buttons on the garment, which can serve both functional and decorative purposes",
            "clothes color": "the dominant and accent colors of the garment, playing a key role in aesthetic appeal and mood",
            "clothes length": "the overall length of the garment, influencing how much of the body is covered and the silhouette it creates",
            "clothes pattern": "the repeated visual motifs or designs on the fabric, ranging from stripes and checks to abstract and complex patterns",
            "clothes shape": "the general outline or form of the garment, reflecting how it fits and drapes on the body",
            "collar shape": "the design and contour of the collar, shaping how the neckline is framed",
            "sleeve length": "the length of the sleeves, determining the coverage of the arms and contributing to the overall style",
            "sleeve shape": "the specific style and contour of the sleeves, affecting how they flow and fit on the arms",

            "texture-related": "the clothing's surface texture, capturing patterns, grain, and tactile appearance, ranging from smooth and uniform to detailed and intricate designs",
            "fabric-related": "the material composition of the garment, highlighting the weave, thickness, and flexibility of fabrics, which influence the garment's feel and drape",
            "shape-related": "the structural outline of the garment, focusing on proportions, symmetry, and how the clothing interacts with the wearer's body shape",
            "part-related": "specific parts of the garment, including sleeves, collars, or hems, emphasizing how individual segments contribute to the overall style",
            "style-related": "the overall stylistic expression of the clothing, encompassing themes such as formal, casual, avant-garde, or classic aesthetics",
        }
        self.indices = {}
        for i, attr in enumerate(self.attrs):
            self.indices[attr] = [[] for _ in range(self.num_values[i])]

        label_file = os.path.join(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, cfg.DATA.GROUNDTRUTH.TRAIN)
        assert os.path.exists(label_file), f"Train label file {label_file} does not exist."
        with open(label_file, 'r') as f:
            for l in f:
                l = [int(i) for i in l.strip().split()]
                fid = l[0]
                attr_val = [(l[i], l[i+1]) for i in range(1, len(l), 2)]
                for attr, val in attr_val:
                    self.indices[self.attrs[attr]][val].append(fid)

    def __len__(self):
        return math.ceil(self.num_triplets / self.batch_size)

    def __str__(self):
        return f"| Triplet Sampler | iters {self.__len__()} | batch size {self.batch_size}|"

    def __iter__(self):
        sampled_attrs = random.choices(range(0, len(self.attrs)), k=self.num_triplets)
        for i in range(self.__len__()):
            attrs = sampled_attrs[i*self.batch_size:(i+1)*self.batch_size]

            anchors = []
            positives = []
            negatives = []
            for a in attrs:
                # Randomly sample two attribute values
                vp, vn = random.sample(range(self.num_values[a]), 2)
                # Randomly sample an anchor image and a positive image
                x, p = random.sample(self.indices[self.attrs[a]][vp], 2)
                # Randomly sample a negative image
                n = random.choice(self.indices[self.attrs[a]][vn])
                # Get the attrKey and attrValue
                attrkey = self.attrs[a]
                prompt = self.attr_desc[attrkey]
                text_p =f"a photo of fashion with focus on {prompt}"
                text_n =f"a photo of fashion with focus on {prompt}"

                anchors.append((x, a, text_p))
                positives.append((p, a, text_p))
                negatives.append((n, a, text_n))

            yield anchors + positives + negatives


class ImageSampler(Sampler):
    def __init__(self, cfg, file):
        self.batch_size = cfg.DATA.TEST_BATCHSIZE
        self.attrs = cfg.DATA.ATTRIBUTES.NAME
        self.attr_desc = {
            "skirt_length": "the skirt's length, affecting the level of leg visibility and overall silhouette, ranging from shorter to longer styles",
            "sleeve_length": "the sleeve length, determining arm coverage and influencing the outfit's overall balance between exposure and coverage",
            "coat_length": "the coat's length, shaping the upper body coverage and overall appearance, with styles that range from shorter cuts to longer, fuller lengths",
            "pant_length": "the length of the pants, impacting leg coverage and silhouette, with variations from shorter cuts to full-length options",
            "collar_design": "the collar's design, which frames the neckline in different ways, offering both traditional and contemporary looks",
            "lapel_design": "the lapel style, shaping the coat's opening and contributing to either a structured or flowing look around the neck and shoulders",
            "neckline_design": "the neckline shape, defining how the neck and shoulders are highlighted, with options from minimalist to more pronounced designs",
            "neck_design": "the neck area's style, affecting how the neck is framed and the balance between openness and coverage around the neck",

            "clothes category": "the general category of the garment, identifying its primary purpose or function, such as outerwear, tops, or bottoms",
            "clothes button": "the presence and arrangement of buttons on the garment, which can serve both functional and decorative purposes",
            "clothes color": "the dominant and accent colors of the garment, playing a key role in aesthetic appeal and mood",
            "clothes length": "the overall length of the garment, influencing how much of the body is covered and the silhouette it creates",
            "clothes pattern": "the repeated visual motifs or designs on the fabric, ranging from stripes and checks to abstract and complex patterns",
            "clothes shape": "the general outline or form of the garment, reflecting how it fits and drapes on the body",
            "collar shape": "the design and contour of the collar, shaping how the neckline is framed",
            "sleeve length": "the length of the sleeves, determining the coverage of the arms and contributing to the overall style",
            "sleeve shape": "the specific style and contour of the sleeves, affecting how they flow and fit on the arms",

            "texture-related": "the clothing's surface texture, capturing patterns, grain, and tactile appearance, ranging from smooth and uniform to detailed and intricate designs",
            "fabric-related": "the material composition of the garment, highlighting the weave, thickness, and flexibility of fabrics, which influence the garment's feel and drape",
            "shape-related": "the structural outline of the garment, focusing on proportions, symmetry, and how the clothing interacts with the wearer's body shape",
            "part-related": "specific parts of the garment, including sleeves, collars, or hems, emphasizing how individual segments contribute to the overall style",
            "style-related": "the overall stylistic expression of the clothing, encompassing themes such as formal, casual, avant-garde, or classic aesthetics",
        }
        self.labels = []

        label_file = os.path.join(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, file)
        assert os.path.exists(label_file), f"Train label file {label_file} does not exist."
        with open(label_file, 'r') as f:
            for l in f:
                l = [int(i) for i in l.strip().split()]
                self.labels.append(tuple(l))

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __str__(self):
        return f"| Image Sampler | iters {self.__len__()} | batch size {self.batch_size}|"

    def __iter__(self):
        for i in range(self.__len__()):
            batch_labels = self.labels[i*self.batch_size:(i+1)*self.batch_size]
            batch = []
            for l in batch_labels:
                fid = l[0]
                attr_val = [(l[i], l[i+1]) for i in range(1, len(l), 2)]
                for attr, val in attr_val:
                    # Get the attrKey and attrValue
                    attrkey = self.attrs[attr]
                    prompt = self.attr_desc[attrkey]
                    text = f"a photo of fashion with focus on {prompt}"
                    batch.append((fid, attr, val, text))
            yield batch
