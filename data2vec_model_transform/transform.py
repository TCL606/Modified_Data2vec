import torch
import argparse

def stu2tea(stu_1_path, stu_2_path, new_path):
    # student_1 is older, to extract the teacher
    # student_2 is newer
    tea_data2vec = torch.load(stu_1_path)
    stu_data2vec = torch.load(stu_2_path)
  
    teacher_dict = tea_data2vec['model']
    teacher_dict_modify = dict()
    for k,v in teacher_dict.items():
        if('encoder.' in k):
            teacher_dict_modify[k[8:]] = v
    stu_data2vec['model']['_ema'] = teacher_dict_modify # set teacher model
    torch.save(stu_data2vec, new_path)
    
def tea2stu(stu_1_path, new_path):
    # use stu1's teacher as student
    data2vec_model = torch.load(stu_1_path)
    teacher_dict = data2vec_model['model']['_ema']
    student_dict = data2vec_model['model']

    for k, _ in student_dict.items():
        if('encoder.' in k):
            student_dict[k] = teacher_dict[k[8:]]
             
    data2vec_model['model'] = student_dict
    torch.save(data2vec_model, new_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='stu2tea', help='either stu2tea or tea2stu')
    parser.add_argument('--stu1', type=str, help='stu1 model path, which provides teacher')
    parser.add_argument('--stu2', type=str, help='stu2 model path, which provides student')
    parser.add_argument('--dest', type=str, help='the destination path of the new model')
    opt = parser.parse_args()
    
    if opt.mode == 'stu2tea':
        stu2tea(opt.stu1, opt.stu2, opt.dest)
    elif opt.mode == 'tea2stu':
        tea2stu(opt.stu1, opt.dest)
    else:
        raise NotImplementedError(f'mode should be stu2tea or tea2stu, but got {opt.mode}')
    print("successfully transformed!")