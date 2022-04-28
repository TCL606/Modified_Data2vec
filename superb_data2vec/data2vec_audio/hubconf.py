from .expert import UpstreamExpert as _UpstreamExpert
import os


def data2vec_audio(*args, **kwargs):
    return data2vec_audio_base(*args, **kwargs)

def data2vec_local(ckpt, *args, **kwargs):
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)

def data2vec_audio_base(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/upstream_model/audio_base_ls.pt"
    return data2vec_local(ckpt=ckpt)

def data2vec_audio_large(*args, **kwargs):
    ckpt="/mnt/lustre/sjtu/home/xc915/superb/upstream_model/vox_pretrained.pt"
    return data2vec_local(ckpt=ckpt)