from collections import OrderedDict
from typing import List, Union, Dict
from s3prl.upstream.interfaces import UpstreamBase
from s3prl.utility.helper import zero_mean_unit_var_norm
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import fairseq

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.name = "data2vec_audio"
        print(f"model: {self.name}")

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt] 
        )
        self.model = model[0]
        self.task = task
        self.wav_normalize = cfg.task.normalize
        self.apply_padding_mask = True

        # These options are only used for aligning representations between s3prl and huggingface
        # See utility/compare_wav2vec2.py
        # self.apply_padding_mask = True
        # self.numpy_wav_normalize = False
        # if len(self.hooks) == 0:
        #     module_name = "self.model.encoder.layers"
        #     for module_id in range(len(eval(module_name))):
        #         self.add_hook(
        #             f"{module_name}[{module_id}]",
        #             lambda input, output: input[0].transpose(0, 1),
        #         )
        #     self.add_hook("self.model.encoder", lambda input, output: output[0])


    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 320

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        device = wavs[0].device
        # if self.wav_normalize:
        #     if self.numpy_wav_normalize:
        #         wavs = zero_mean_unit_var_norm([wav.cpu().numpy() for wav in wavs])
        #         wavs = [torch.from_numpy(wav).to(device) for wav in wavs]
        #     else:
        wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.model.extract_features(
            padded_wav, wav_padding_mask if self.apply_padding_mask else None
        )
        
        feature = []
        feature.append(results['x'])
        for lr in results['layer_results']:
            feature.append(lr[2].transpose(0, 1))
        return {
            "hidden_states": feature,
            "PR": feature,
            "ASR": feature,
            "QbE": feature,
            "SID": feature,
            "ASV": feature,
            "SD": feature,
            "ER": feature,
            "SF": feature,
            "SE": feature,
            "SS": feature,
            "secret": feature,
        }
