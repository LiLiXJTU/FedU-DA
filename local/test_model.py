import numpy as np
import torch.optim
import torch.nn as nn
from scipy.ndimage import _ni_support, generate_binary_structure, binary_erosion, distance_transform_edt
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
import warnings
import numpy

from local.UfNet import UCTransNet
from local.fusiontrans import UFusionNet
from src.LoadData import ValGenerator, ImageToImage2D
from src.models import UNet

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2

def show_image_with_dice(predict_save, labs,names,save_path):
    predict_save = np.argmax(predict_save,axis=1)
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    img_pre = np.reshape(predict_save, (lab.shape[1], lab.shape[2])) * 255
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
    return dice_pred
def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path,names, dice_pred, dice_ens):
    model.eval()
    output = model(input_img.cuda())
    outputsoft = torch.softmax(output,dim=1).cpu().data.numpy()
    # pred_class = torch.where(outputsoft>0.5,torch.ones_like(outputsoft),torch.zeros_like(outputsoft))
    # predict_save = pred_class[0].cpu().data.numpy()
    # predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp = show_image_with_dice(outputsoft, labs,names,save_path=vis_save_path+'_predict'+model_type+'.jpg')

    #dice_one = metric.binary.dc(pre, labs)
    #sen_one = metric.binary.sensitivity(pre, labs)
    #hd_one = hd95(pre, labs, voxelspacing=None, connectivity=1)
    return dice_pred_tmp



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    if config.task_name is "CTseg":
        test_num = 470
        model_type = config.model_name
        model_path = "C:\code/fedsemi-l/fedbestmodels/1/best_model-1.pth.tar"

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
        model = UNet(model_type,3,2)
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
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    dice = 0.0
    sensitivity = 0.0
    hd_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path+names[0][:-4]+"_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+str(i),names,
                                               dice_pred=dice_pred, dice_ens=dice_ens)
            #hd_t = metric.binary.hd95(input_img, lab)
            dice_pred+=dice_pred_t
            # dice+=dice_one
            # sensitivity+=sen_one
            #hd_pred+= hd_one
            torch.cuda.empty_cache()
            pbar.update()
    print ("dice_pred",dice_pred/test_num)
    # print('dice',dice/test_num)
    # print('sensitivity', sensitivity / test_num)
    #print('hd', hd_pred / test_num)



