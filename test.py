import numpy as np
import torch.optim
import torch.nn as nn
from PIL import Image
from scipy.ndimage import _ni_support, generate_binary_structure, binary_erosion, distance_transform_edt
from sklearn.metrics import jaccard_score
from sklearn import metrics
from torch.utils.data import DataLoader
import warnings
import numpy
import numpy as np
from medpy.metric.binary import dc

from local.UfNet import UCTransNet
from local.fusiontrans import UFusionNet
from src.LoadData import ValGenerator, TestImageToImage2D
from src.models import UNet
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
# def one_hot_encoder(x, num_classes, on_value=1., off_value=0.):
#     x = x.unsqueeze(1)
#     return torch.full((x.size()[0], num_classes,x.size()[2],x.size()[2]), off_value).scatter_(1, x, on_value)
class BinaryDice(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDice, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target, flag):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = self.smooth
        union = self.smooth
        if flag is None:
            pd = predict
            gt = target
            intersection += torch.sum(pd*gt)*2
            union += torch.sum(pd.pow(self.p) + gt.pow(self.p))
        else:
            for i in range(target.shape[0]):
                if flag[i,0] > 0:
                    pd = predict[i:i+1,:]
                    gt = target[i:i+1,:]
                    intersection += torch.sum(pd*gt)*2
                    union += torch.sum(pd.pow(self.p) + gt.pow(self.p))
        dice = intersection / union

        return dice

class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, input, target):

        #N = target.size(0)
        N = target.shape[0]
        smooth = 1e-5

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        dice = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_mean = dice.mean()

        return dice_mean


class MultiClassDice(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MultiClassDice, self).__init__()

    def forward(self, input, target, weights=None):
        #可取当前mask所代表的值
        C = input.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        #可取当前mask所代表的值
        #target[target > 0].data[0]
        client_id = 0
        if client_id == 0:
            class_flag = [0,1]
        if client_id == 1:
            class_flag = [2]
        if client_id == 2:
            class_flag = [3]
        dice = BinaryDice()
        totalDice = []

        target = torch.nn.functional.one_hot(torch.from_numpy(target), C).reshape(1, C, 224, 224)
        input = torch.from_numpy(input)
        #target = target.numpy()
        #target = one_hot_encoder(target, C)

        for i in range(C):
            if i in class_flag:
                metricDice = dice(input[:, i], target[:, i], flag = None)
                dice_one = dc(input[:, i], target[:, i])
                if weights is not None:
                    metricDice *= weights[i]
                totalDice.append(metricDice.item())
        print(totalDice)

        return sum(totalDice)/C
def show_image_with_dice(predict_save, labs,names,save_path):
    label = labs[0]

    #dice_pred = MultiClassDice()(predict_save,labs)

    predict_save = torch.max(torch.from_numpy(predict_save), dim=1).indices
    predict_save = predict_save.numpy()
    pred = predict_save[0]
    #dice_pred = MultiClassDice()(predict_save, labs)
    dice = []
    one_pred = np.zeros_like(pred)
    two_pred = np.zeros_like(pred)
    thr_pred = np.zeros_like(pred)
    f_pred = np.zeros_like(pred)

    one_lab = np.zeros_like(pred)
    two_lab = np.zeros_like(pred)
    thr_lab = np.zeros_like(pred)
    f_lab = np.zeros_like(pred)

    for i in range(5):
        if i == 1:
            one_lab[label==1] = 1
            one_pred[pred==1] = 1
            dice_pred = dc(one_pred, one_lab)
            dice.append(dice_pred)
        elif i == 2:
            two_pred[pred==2]=1
            two_lab[label==2]=1
            dice_pred = dc(two_pred, two_lab)
            dice.append(dice_pred)
        elif i == 3:
            thr_pred[pred==3]=1
            thr_lab[label==3] = 1
            dice_pred = dc(thr_pred, thr_lab)
            dice.append(dice_pred)
        elif i == 4:
            f_pred[pred==4]=1
            f_lab[label==4] = 1
            dice_pred = dc(f_pred, f_lab)
            dice.append(dice_pred)

    print(dice,'dice')
    img_pre = np.reshape(predict_save, (lab.shape[1], lab.shape[2]))
    fig, ax = plt.subplots()
    plt.imshow(img_pre, cmap='gray')
    plt.axis("off")
    height, width = config.img_size, config.img_size
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(vis_path + names[0][:-4] + "_pre.jpg", dpi=300)
    plt.close()
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    # predict_save[predict_save > 0] = 255
    # predict_save[predict_save <= 0] = 0
    # labs[labs > 0] = 255
    # labs[labs <= 0] = 0
    #
    # cv2.imwrite(vis_path + names[0][:-4] + "_gttest.jpg", labs)
    # cv2.imwrite(vis_path + names[0][:-4] + "_predtest.jpg", predict_save)
    return dice
def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path,names, dice_pred, dice_ens):
    model.eval()
    output = model(input_img.cuda())
    outputsoft = torch.softmax(output,dim=1).cpu().data.numpy()
    #outputsoft = torch.softmax(output, dim=1)
    #outputsoft = torch.max(outputsoft,dim=1).indices.cpu().data.numpy()
    # pred_class = torch.where(outputsoft>0.5,torch.ones_like(outputsoft),torch.zeros_like(outputsoft))
    # predict_save = pred_class[0].cpu().data.numpy()
    # predict_save = np.reshape(predict_save, (config.img_size, config.img_size))

    mask = torch.argmax(output, dim=1)
    # outputs = outputs.data.max(1)[1]
    # outputs = torch.max(outputs, dim=1).values.unsqueeze(1)
    # outputs = torch.argmax(outputs, dim=1)
    img_path = './Synapse_3_full_our/'
    for i in range(labs.shape[0]):
        labels_arr = labs[i]
        mask_arr = mask[i].cpu().numpy()
        # 定义颜色映射表
        color_map = {
            0: (0, 0, 0),  # 类别0为黑色
            1: (255, 150, 113),  # 类别1为橙色
            2: (132, 94, 194),  # 类别2为紫色
            3: (53, 150, 181),  # 类别3为蓝色
            # 4: (206, 62, 43),  # 类别4为红色
            4: (0, 255, 0),  # 类别4为红色
            5: (128, 0, 128),  # 类别5为深紫色
            6: (0, 255, 255),  # 类别6为青色
            7: (128, 128, 128),  # 类别7为灰色
            8: (255, 0, 0),  # 类别8为红色
            9: (0, 255, 0),  # 类别9为绿色
            10: (255, 255, 0),  # 类别10为黄色
            11: (0, 0, 255),  # 类别11为蓝色
        }

        # 将像素值映射成颜色
        colors = np.zeros((*mask_arr.shape, 3), dtype=np.uint8)
        for j, color in color_map.items():
            colors[mask_arr == j, :] = color

        # 将颜色数组保存为图片
        # img = Image.new('RGB', (colors.shape[1], colors.shape[0]))
        # data = colors.reshape(-1, colors.shape[2]).tolist()
        # img.putdata(data)
        # img.save(img_path + labelName[i][:-4] + ".png")
        img = Image.fromarray(colors)
        img.save(img_path + names[i][:-4] +'_premask'+ ".png")

        # 将像素值映射成颜色
        gt_colors = np.zeros((*labels_arr.shape, 3), dtype=np.uint8)
        for j, color in color_map.items():
            gt_colors[labels_arr == j, :] = color

        img = Image.fromarray(gt_colors)
        img.save(img_path + names[i][:-4] + "_gt_new.png")


    dice_pred_tmp = show_image_with_dice(outputsoft, labs,names,save_path=vis_save_path+'_predict'+model_type+'.jpg')

    #dice_one = metric.binary.dc(pre, labs)
    #sen_one = metric.binary.sensitivity(pre, labs)
    #hd_one = hd95(pre, labs, voxelspacing=None, connectivity=1)
    return dice_pred_tmp



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    if config.task_name is "ALL":
        test_num = 71
        model_type = config.model_name
        #model_path = "D:/quanzhong/Fed/1234/300slices/unetse_focal_8_1e-5/global_model/26/best_model-UNet.pth.tar"
        #model_path = "D:/quanzhong/Fed/1234/300slices/114/unetsk_focal_8-1e-5\global_model/15/best_model-UNet.pth.tar"
        #model_path ='C:/li/Fed/fedsemi-l/global_model/17/best_model-UNet.pth.tar'
        # model_path = "D:/quanzhong/Fed/1234/300slices/unet_focal_8_1e-5/global_model/29/best_model-UNet.pth.tar"
        #model_path = r"D:\quanzhong\Fed\1234\300slices\quebiaoqian\2client\BC\global_model\29/best_model-UNet.pth.tar"
        #model_path=r'D:\quanzhong\Fed\1234\300slices\quebiaoqian\0_34\EMA\global_model\26\best_model-UNet.pth.tar'
        #model_path = r'D:\quanzhong\Fed\1234\300slices\quebiaoqian\2_34\EMA\global_model\27\best_model-UNet.pth.tar'
        #model_path = 'D:/quanzhong/hunhe/1234/300slices/unet_focal_8_1e-5/pretain_path/149/best_model-UNet.pth.tar'
        #model_path = 'D:/quanzhong/hunhe/1234/300slices/UNet/80/best_model-UNet.pth.tar'
        #model_path = 'D:/quanzhong/pretrain/A/pretain_path/149/best_model-UNet.pth.tar'

        #3_full_our
        model_path = r'D:/quanzhong/Fed/1234/300slices/114/unetsk_focal_8-1e-5\global_model/15/best_model-UNet.pth.tar'
        #3_full_centra
        #model_path = 'D:/quanzhong/hunhe/1234/300slices/unet_focal_8_1e-5/pretain_path/149/best_model-UNet.pth.tar'
        #3_full_fedunet
        #model_path = "D:/quanzhong/Fed/1234/300slices/unet_focal_8_1e-5/global_model/29/best_model-UNet.pth.tar"


    elif config.task_name is "MoNuSeg":
        test_num = 14
        model_type = config.model_name
        model_path = "./MoNuSeg/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"


    save_path  = config.task_name +'/'+ model_type +'/' + test_session + '/'
    vis_path = "./" + '22' + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')


    if model_type == 'UNet':
        config_vit = config.get_CTranS_config()
        # model = UNet_3Plus_DeepSup_CGM(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)
        model = UNet(config.model_name,1,5)
    elif model_type == 'UFusionNet':
        config_vit = config.get_CTranS_config()
        # model = UCTransNet3Plus(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)
        model = UFusionNet(model_type,3,2)

    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    #test_dataset = ImageToImage2D(test_path, tf_test, image_size=config.img_size)
    test_dataset = TestImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = [0,0,0,0]
    dice = 0.0
    sensitivity = 0.0
    hd_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            print(names)
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2]))
            fig, ax = plt.subplots()
            plt.imshow(arr[0,0,:,:], cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig('./d_unet_fed/'+names[0][:-4]+"_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+str(i),names,
                                               dice_pred=dice_pred, dice_ens=dice_ens)
            #hd_t = metric.binary.hd95(input_img, lab)
            for j in range(4):
                dice_pred[j] = dice_pred[j]+dice_pred_t[j]
            # dice+=dice_one
            # sensitivity+=sen_one
            #hd_pred+= hd_one
            torch.cuda.empty_cache()
            pbar.update()
            dice_sum = dice_pred[0]+dice_pred[1]+dice_pred[2]+dice_pred[3]
    print ("0_dice_pred",dice_pred[0]/test_num)
    print("1_dice_pred", dice_pred[1] / test_num)
    print("2_dice_pred", dice_pred[2] / test_num)
    print("3_dice_pred", dice_pred[3] / test_num)
    print("avg_dice_pred", dice_sum / test_num/4)
    # print('dice',dice/test_num)
    # print('sensitivity', sensitivity / test_num)
    #print('hd', hd_pred / test_num)



