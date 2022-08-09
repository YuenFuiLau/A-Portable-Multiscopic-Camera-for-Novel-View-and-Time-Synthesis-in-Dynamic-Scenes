from model import *
import os
import os.path
import numpy as np
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
import imageio


#load image
def load_imgs(image_dir):

    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,std=std)

    transform = transforms.Compose([transforms.ToTensor(), normalize])
   
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names
    img_paths = [os.path.join(image_dir, n) for n in img_names]
    N_imgs = len(img_paths)

    img_list = []
    for p in img_paths:
        img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
        img = PILImage.fromarray(img).resize((512,384),PILImage.BILINEAR) #reshape (640,360)
        img = transform(img)  # (3,H, W) 
        img_list.append(img)

    img_list = torch.stack(img_list,dim=0) #(N, 3, H, W) torch.float32

    H, W = img_list.shape[2], img_list.shape[3]
    
    results = {
        'imgs': img_list,  #(N, 3, H, W) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'H': H,
        'W': W,
    }
    return results

#load image data

def load_img_data(image_dir):

    folders = os.listdir(image_dir)
    folders.sort(key = lambda x: int(x))
    TSteps = len(folders)

    results = {}
    images_data = {}
    
    for i in range(TSteps):

        path = f"{image_dir}/{folders[i]}/images"
        
        image_info = load_imgs(path) 
        imgs = image_info['imgs']      #(N, 3, H, W) torch.float32

        #save imgs pack
        images_data[i] = imgs
        
        N_IMGS = image_info['N_imgs']
        H = image_info['H']
        W = image_info['W']

    results = {
        'images_data': images_data,  # dict
        'TSteps':TSteps,
        'N_IMGS': N_IMGS,
        'H': H,
        'W': W,
    }

    return results

def render_inp_frame(frame0,frame1,intermediateIndex,num_inp_frame,flowComp,ArbTimeFlowIntrp,flowBackWarp):

    #set up
    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)
    
    TP = transforms.Compose([revNormalize,transforms.ToPILImage()])

    with torch.no_grad():

        I0 = frame0
        I1 = frame1

        flowOut = flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:,:2,:,:]
        F_1_0 = flowOut[:,2:,:,:]

        # Generate intermediate frames
        t = float(intermediateIndex) / num_inp_frame
        temp = -t * (1 - t)
        fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
        g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1   = 1 - V_t_0

        g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
        g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

        wCoeff = [1 - t, t]

        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1) #(N,3,H,W)

        # Save intermediate frame
        N_CAMS = Ft_p.shape[0]
        result = []
        for batchIndex in range(N_CAMS):
            result.append( TP(Ft_p[batchIndex]) )

        return torch.from_numpy( np.stack(result,axis=0) ).float() #(N,H,W,3)
    
def flow_inp_frame(frame0,frame1,intermediateIndex,num_inp_frame,flowComp,ArbTimeFlowIntrp,flowBackWarp):

    #set up
    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)
    
    TP = transforms.Compose([revNormalize,transforms.ToPILImage()])

    with torch.no_grad():

        I0 = frame0
        I1 = frame1

        flowOut = flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:,:2,:,:]
        F_1_0 = flowOut[:,2:,:,:]

        # Generate intermediate frames
        t = 1. / 2
        temp = -t * (1 - t)
        fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
        g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1   = 1 - V_t_0

        g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
        g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

        wCoeff = [1 - t, t]

        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1) #(N,3,H,W)

        # Save intermediate frame
        N_CAMS = Ft_p.shape[0]
        result = []
        for batchIndex in range(N_CAMS):
            result.append( TP(Ft_p[batchIndex].cpu()) )

        return torch.from_numpy( np.stack(result,axis=0) ).float() #(N,H,W,3)

        # return V_t_0, V_t_1, g_I0_F_t_0_f, g_I1_F_t_1_f, wCoeff
    


def main():

    scene_name = "room6"
    extractionPath = f"{os.getcwd()}/data/{scene_name}"
    outputPath     = f"{os.getcwd()}/nvs_result"
    ckpt_path = f"{os.getcwd()}/ckpt/SuperSloMo.ckpt"

    #load data
    dataset_info = load_img_data(extractionPath)

    images_data = dataset_info['images_data']
    TSteps = dataset_info['TSteps']    
    N_IMGS = dataset_info['N_IMGS']
    H = dataset_info['H']
    W = dataset_info['W']


    # Initialize transforms
    device = torch.device("cuda:9")

    # Initialize model
    flowComp = UNet(6, 4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    ArbTimeFlowIntrp = UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    flowBackWarp = backWarp(W, H, device)
    flowBackWarp = flowBackWarp.to(device)

    dict1 = torch.load(ckpt_path, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    # Interpolate frames
    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)
    TP = transforms.Compose([revNormalize,transforms.ToPILImage()])

    result = [[] for i in range(N_IMGS)]
    num_of_inp_frame = 4
    for t in range(TSteps-1):

        frame0 = images_data[t]
        frame0 = frame0.to(device)
        frame1 = images_data[t+1]
        frame1 = frame1.to(device)

        for i in range(N_IMGS):
            
            img = torch.from_numpy( np.array( TP(frame0[i]) ) )#(H, W, 3)
            result[i].append(img)

        for inp_idx in range(1,num_of_inp_frame):

            inp_imgs = render_inp_frame(frame0 = frame0,
                                        frame1 = frame1,
                                        intermediateIndex = inp_idx,
                                        num_inp_frame = num_of_inp_frame,
                                        flowComp = flowComp,
                                        ArbTimeFlowIntrp = ArbTimeFlowIntrp,
                                        flowBackWarp = flowBackWarp,
                                        )

            for i in range(N_IMGS):
            
                img =  inp_imgs[i] #(H, W, 3)
                result[i].append(img)

    for i in range(N_IMGS):
        
        video = torch.stack( result[i] , dim=0 ).cpu().numpy().astype(np.uint8)
        imageio.mimwrite(os.path.join(outputPath,scene_name+f"SlMo_CAMS{i}.gif"),video,fps=30)


if __name__ == "__main__":

    main()