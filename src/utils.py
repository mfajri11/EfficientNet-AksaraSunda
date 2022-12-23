import pathlib
import torchvision
import torch
import torch.nn as nn
import splitfolders
import os
import re
from torchinfo import summary
from torch.utils.data import DataLoader,ConcatDataset
from torchvision import datasets, transforms
from torchvision.ops import Conv2dNormActivation, SqueezeExcitation
from torchvision.models.efficientnet import MBConv
from typing import Any, Sequence, Union, Tuple, List
from torchvision.models import EfficientNet
import matplotlib.pyplot as plt
from random import sample
from PIL import Image


def _get_weights_moduls_count(children: Sequence[nn.Module], kind: str = 'conv2d') -> Tuple[
                                 List[nn.parameter.Parameter],
                                 List[nn.Module],
                                 int]:
    """retun tuple of list of torch Module, torch paramater and number of nodes
    by node means convolution/pooling/activation function
    
    Parameters
    ----------
    children : List[torch.nn.Module]
        child of model to find wanted nodes
    mode : str
        the type of nodes we want supported [conv2d, pooling, activation, batchnorm] (default: conv2d)
    
    Examaple
    --------
    weights, nodes, count = _get_weights_moduls_count(children=children, mode=kind)
    """
    kind = kind.lower()
    weights = []
    moduls = []
    counter = 0
    container_like = (nn.Sequential, Conv2dNormActivation, MBConv, SqueezeExcitation)
    nn_moduls = {
        'conv2d': nn.Conv2d, 'kernel': nn.Conv2d,
        'activation': [nn.SiLU, nn.Sigmoid, nn.ReLU], 
        'pooling': nn.AdaptiveAvgPool2d, 'batchnorm':nn.BatchNorm2d}
    want_node = nn_moduls[kind]
    
    def helper(children: Sequence[nn.Module], 
               weights: List[nn.parameter.Parameter], 
               moduls: List[nn.Module],
               counter: int)-> Tuple[
                                 List[nn.parameter.Parameter],
                                 List[nn.Module],
                                 int]:
                   
        for child in children:
            is_same = ((type(child) in want_node) 
                       if type(want_node) == list 
                       else (type(child) == want_node))
            if is_same:
                counter += 1
                moduls.append(child)
                if kind == 'conv2d' or kind == 'kernel':
                    weights.append(child.weight)
            elif type(child) in container_like:
                weights, moduls, counter = helper(child.children(), weights, moduls, counter)
        return weights, moduls, counter
    
    return helper(children, weights, moduls, counter)
    
def get_image_with_transform(path: str, 
                transform_func: transforms.Compose, 
                device: torch.device = torch.device('cpu')) -> torch.Tensor:
    
    """return torch Tensor represent an image
    
    Parameters
    ----------
    
    path : str
        path or location of image
    transform_func : torchvision.transform.Compose
        transformation function that would be applied to the image (default: None)
    device : torch.device
        the device to locate the image could be CPU or GPU (default: CPU)
        
    Example
    -------
    import utils
    
    utils.get_image_with_transform(path='path/to/image')"""
    
    img = Image.open(path)
    img = transforms.PILToTensor()(img)
    img = transform_func(img=img)
    img = img.unsqueeze(0)
    img = img.to(device)
    return img


def _show_image(images, names):
    # fig = plt.figure(figsize=(30, 50))
    # for i in range(len(images)):
    #     a = fig.add_subplot(5, 4, i+1)
    #     imgplot = plt.imshow(images[i])
    #     a.axis("off")
    #     a.set_title(names[i].split('(')[0], fontsize=30)
    
    
    for num_layer in range(len(images)):
        plt.figure(figsize=(50, 10))
        layer_viz = images[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print("Layer ",num_layer+1)
        for i, filter in enumerate(layer_viz):
            if i == 16: 
                break
            plt.subplot(2, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
            plt.title(names[i].split('(')[0])
        # plt.show()
        # plt.close()
        plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')

def get_EfficientNet(device=torch.device('cpu'))-> EfficientNet:
    """return a EfficientNet(-B0) instance using IMAGENET Weight
    
    Parameter
    ---------
    
    device: torch.device
        torch device that would be used to locate the data whether it would be on CPU or GPU (default: CPU)

    Example
    -------
    import utils
    
    utils.get_EfficientNet()"""
    
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    return model

def get_EfficientNet_transform() -> transforms.Compose:
    """return specific transformation function for EfficientNet-B0 
    
    Example
    --------
    import utils
    
    utils.get_EfficientNet_transform()"""
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    tf = weights.transforms()
    return tf
    

def is_allowed_path_type(path: Union[str, pathlib.Path]) -> bool:
    """return true if path is instance of str or pathlike
    
    parameters
    ----------
    path : str
        path or location of file/directory
    
    Example
    -------
    import utils
    
    utils.is_allowed_path_type('path/to/test')"""
    
    return isinstance(path, pathlib.Path) or isinstance(path, str)

def convert_str_to_path(path: Union[str, pathlib.Path]) -> pathlib.Path:
    if not is_allowed_path_type(path):
        raise Exception("path type must be string or pathlib.Path")
    if isinstance(path, pathlib.Path):
        return path
    return pathlib.Path(path)

def create_data_loader(train_path: Union[str, pathlib.Path],
                       test_path: Union[str, pathlib.Path],
                       transform_func: transforms.Compose,
                       num_workers: int = 2,
                       batch_size: Union[int, None] = None,
                       augmentation_func: transforms.Compose = None
                       ) -> Tuple[DataLoader, DataLoader, List[str]]:
    
    """return a tuple of (pytorch Dataloader of training, pytorch Dataloader of testing, list of its classes)
    this function provide a helper for creating dataloader without worrying create a torch dataset object 
    
    Parameters
    ----------
    
    train_path : str, pathlike
        path or location of training data
    test_path : str, pathlike
        path or location of test data
    transform_func : torchvision.transform.Compose
        function for transforming the data basead on the Compose args
    num_workers : int
        number of subproces wanted for loading the data to RAM
    batch_size : int
        number of sample for each iteration / batch
    transform_func : torchvision.transform.Compose (default: None)
        transformation function for augmenting the data
    
    
    Example
    -------
    import utils
    from torchvision import transorm
    
    trainloader, testloader, classes = utils.create_data_loader(
        train_path='path/to/train/file',
        test_path = 'path/to/test/file',
        num_workers = 2,
        batch_size = 64,
        transform_func = transform.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])
    )
    """
    
    train_path = convert_str_to_path(train_path)
    test_path = convert_str_to_path(test_path)
    trainset = datasets.ImageFolder(str(train_path), transform=transform_func)
    testset = datasets.ImageFolder(str(test_path), transform=transform_func)
    classes = trainset.classes
    
    if augmentation_func is not None:
        testset_tmp = testset
        trainset_tmp = trainset
        
        trainset = ConcatDataset(datasets=[
            datasets.ImageFolder(
                root=train_path,
                transform=augmentation_func),
            trainset_tmp
        ])
        
        testset = ConcatDataset(datasets=[
            datasets.ImageFolder(
                root=str(test_path),
                transform=augmentation_func
            ),
            testset_tmp
        ])
    
    train_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
    # pin_memory=True    
    )

    test_loader = DataLoader(
    testset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=num_workers
    # pin_memory=True
    )
    
    return train_loader, test_loader, classes
    


    

def split_image_folder(path: str, target_path:str, ratio: str) -> None:
    """this function provide helper for splitting image folder to train/testing with certain ratio
    this function will produce side effect and return None
    the image folder structure must follow this structure:
        required image folder structure
        data
          |------ class_name
          |------ another_class_name
          | .
          | .
          | .
          |------ others_class_name
          
        folder structure after applied by function
        <taget_path>/<train_ratio>_<test_ratio>
            |------test
            |       |--- class_name
            |       |--- another_class_name
            |       | .
            |       | .
            |       | .
            |       |--- others_class_name
            |       
            |------train
            |       |--- class_name
            |       |--- another_class_name
            |       | .
            |       | .
            |       | .
            |       |--- others_class_name
            
        where <target_path> : argument from target_path paramter
              <train_ratio>_<test_ratio> : argument from ratio paramter while "/" replaced with "_" 
    
    Parameters
    ----------
    
    path : str
        location or path of base image folder (folder which contain classes)
    target_path: str
        location or path of parent output result the actual results are in folder ratio
        example
        target_path = "output"
        ratio = "60/40"
        the results file would be on "./output/60_40"
    ratio: str
        ratio for image would be splitted example "60/40"
        
    Example
    --------
    utils.split_image_folder(path='path/to/image', ratio='60/40')
    
    """
    # convert ratio from string to float
    # ratio = '60/40' -> 0.6 (training), 0.4 (testing)
    tr, vr = ratio.split('/') 
    ftr, fvr, = (float(tr) / 100), (float(vr) / 100)
    splitfolders.ratio(path, output=f"{target_path}/{tr}_{vr}",
    seed=1337, ratio=(ftr, fvr), group_prefix=None, move=False)
    
def generate_intermediate_image(image_path: Union[str, pathlib.Path], 
                            model: nn.Module,
                            which_layer : int = 8, 
                            transform_func: transforms.Compose = None,
                            kind: str = 'conv2d',
                            device: torch.device = torch.device('cpu'),
                            cmap: str = 'viridis') -> None:
    """this function will create image applied to nodes like conv2d, pooling, activation function or batch normalization
    
    Parameters
    ----------
    
    image_path : str, pathlike
        path of image which would be applied to nodes
    model : torch.nn.Module
        torch Module to extract nodes used for produce output image
    which_layer : int
        what layer to used to perform its kind operation, if -1 is set all layers are used
    transform_func : torch.transforms.Compose
        transformation function which applied to image before image applied to nodes (default: None)
    kind : str
        the type of nodes we want, supported [conv2d, pooling, activation, batchnorm] (default: conv2d)
    device : torch.device
        the device to locate the image could be CPU or GPU (default: CPU)
    cmap : str
        colormap for coloring the output image
        
    Example
    -------
    import utils
    
    utils.generate_intermediate_image('path/to/image.jpg', model, transform_func, 'conv2d')  
    """
    
    img = get_image_with_transform(image_path, transform_func, device)
    
    weights = []
    convs = []
    children = list(model.children())
    count = 0
    
    weights, convs, count = _get_weights_moduls_count(children=children, kind=kind)
    
    this_weight = weights[which_layer]
    if kind == 'kernel':
        plt.figure()
        for i, weight in enumerate(this_weight):
            if i == 63:
                break
            plt.subplot(8, 8, i+1)
            plt.imshow(weight.detach().cpu().numpy(), cmap=cmap)
            plt.axis('off')
        return
            
        
    
    # apply the image to  convs/activation/batch/pooling
    print(f"total {kind}: {count}")
    res = [convs[0](img)]
    for i in range(1, len(convs)):
        res.append(convs[i](res[-1]))
    outputs = res
    
    layer_viz = outputs[which_layer][0, :, :, :]
    layer_viz = layer_viz.data
    for i, fmap in enumerate(layer_viz):
        plt.figure()
        if i == 63: 
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(fmap.detach().cpu().numpy(), cmap=cmap)
        plt.axis("off")
    # plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
    plt.show()
    
    
def imshow_labels(base: str, N: int = 4) -> None:
    """this function will show N images for every labels.
    the base directory must follow this structure:
        
        example.
        basedir
          |------ class_1
          |------ class_2
          |        |------ file_1.jpg
          |        |------ file_2.jpg
          | .      | .
          | .      | .
          | .      | .
          |        |------ file_m.jgp
          | 
          | 
          | 
          |------ class_n
    
    the name of class folder/directory and its file are valid python string.
    for now this function will break if class more than 18 and should be refactored "soon".
    
    parameters
    ----------
    
    base : str
        the path of directory which contain class directory
    N : int
        number/sample for every class would be showed
    
    Example
    -------
    
    utils.imshow_labels(base='path/to/basedir', N=4)
    """
    
    if  1 <= N >= 4 and not isinstance(N, int):
        raise Exception(f"n must be integer in range [1, 4], got n: {N}")
    
    data: List[str, List[str], List[str]] = list(os.walk(base)) # walk to base dir
    
    # based on list returned by os.walk the labels are positioned on index 0,1
    # the possible shape of list returned by os.walk at this function is (n+1, 3)
    # the first, second and thrid column of returned list 
    # represent as its/root path, list of its sub dir and its files respectively
    # based on that pattern the image file will start at index 1,2 to n,2
    
    labels: List[str] = data[0][1] #get the labels
    impaths = [] # list for fullpath of image files
    
    # get n sample for every labels and add to impaths
    for i, label in enumerate(labels):
        for fname in sample(data[i+1][2], N):
            fpname = base + "/" + label + "/" + fname # create full path image source 
            impaths.append(fpname)
        
    # show every image we have on impaths
    plt.figure(figsize=(9, 9)) # size of figure the number from trial and error
    for i, path in enumerate(impaths):
        plt.subplot(9, 8, i+1) # idem
        img = plt.imread(path)
        plt.imshow(img)
        label = path.split('/')[-2]# every images name are dir/label/<name_file>.jpg
        plt.title(label) # the labels are at the second last index
        plt.axis('off')
        
def get_device():
    """Return Available torch device
    
    Example
    --------
    import utils
    
    utils.get_device()"""
    return torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def get_optimizer(params: torch.nn.parameter.Parameter, name: str = 'Adam', lr: int=1e-5, weight_decay: int =1e-5) -> nn.Module:
    """Return torch optimizer based on given name parameter
    
    Parameter
    ---------
    name : str
        name of optimizer we want (default: Adam)
    
    Example
    -------
    import utils
    
    utils.get_optimizer(name='Adam')
    """
    name = name.lower()
    if name == 'adam':
        return torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif name == 'nadam':
        return torch.optim.NAdam(params=params, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsproop':
        return torch.optim.RMSprop(params=params, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception(f"unsupported optimzer, list of supported optimizer ['Adam', 'Nadam', 'RMSProop'] got: {name}")
    

def imshow_before_after(image_path: str, transform_func : transforms.Compose) -> None:
    """Show  original image in given path and transformed image given a transform function in one figure
    
    Parameters
    ----------
    
    image_path : str
        the location/path of the image
    transform_func : torvision.transforms.Compose
        transformation function to be applied to given image in `image_path`
        
    Example
    -------
    import utils
    
    # assue these arguments are valid
    utils.imshow_before_after('path/to/image.jpg', transform_func)
    """
    
    before = Image.open(image_path)
    after = get_image_with_transform(image_path, transform_func)
    after = after.squeeze(0)
    after = transforms.ToPILImage()(after)
    # after = after.permute(1, 2, 0)
    # after = after.numpy()
    
    
    plt.figure()
    
    for index, (title, image) in enumerate(zip(('before', 'after'), (before, after)), start=1):
    
        plt.subplot(1,2, index)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"{title} {image.size[0]}X{image.size[1]}")
    
    plt.show()

  
def get_summary(model : nn.Module, batch_size : int, output_size : int)-> None:
  """Provide wrapper for summarize given Pytorch model using torchinfo.summary function and add coloring for author's final project
  see https://github.com/TylerYep/torchinfo for more detail
  
  Parameters
  ----------
  
  model : torch.nn.Module
      Pytorch model which want to be summarized
  batch_size : int
      batch sized used
  output_size : int
      output size of model used to be summarized4
  
  Example
  -------
  import utils
  
  utils.get_summary(model=model, batch_size=64, output_size=18) # assume arguments are valid
  """
  recap = (summary(model, 
          input_size=(batch_size, 3, 224, output_size), 
          verbose=0, 
          col_names=['input_size', 'output_size', 'num_params', 'trainable'],
          col_width=18,
          depth=4))
  
  
  red = '\033[91m'
  default = '\033[39m'
  # nt is Non-trainable
  nt = recap.total_params - recap.trainable_params
  nt_pattern = '\nNon-trainable params: [0-9,]+\n'
  nt_subtitution = f'\nNon-trainable params: {red}{nt:,}{default}\n'
  colored_recap = re.sub(nt_pattern, nt_subtitution, str(recap))
  print(colored_recap)