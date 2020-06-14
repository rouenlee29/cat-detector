import os
from PIL import Image
from fastai.vision import *

def identify_bad_images(filenames):
    bad_images = []

    for filename in filenames:
        try:
            img = Image.open(filename) # open the image file
            img.verify() # verify that it is, in fact an image
            exif_data = img._getexif() # to eliminate those with corrupt exif data 
        except:
            bad_images.append(filename)
    return bad_images
            
def remove_bad_files(bad_images):
    for b in bad_images:
        try:
            os.remove(b)
            print(f"File {b} Removed!")
        except:
            pass
        

def get_folder_counts(Dir = "./images", folders_to_exclude = ['models'], folders_to_include = None):
    
    
    if folders_to_include is None:
        
        # count the number of images in each folder 
        subfolders = []
        for root, dirs, files in os.walk(Dir, topdown=False):
            subfolders.append(root)
        # one cat breed = one folder
        # count number of images in each folder 
        cat_breeds = []
        counts = []
        folders = subfolders[:-1]
        
        for f in folders_to_exclude:
            try:
                folders.remove(f'{Dir}\\{f}')
            except ValueError:
                pass
    else:
        folders = [f'{Dir}\\{f}' for f in folders_to_include]

    cat_breeds = []
    counts = []
    for d in folders:
        for _,_,file in os.walk(d):
            cat_name = d.split("\\")[1]
            cat_breeds.append(cat_name)
            counts.append(len(file))
            
    df = pd.DataFrame({"cat breed" : cat_breeds, "count" : counts})
    df.sort_values("count", ascending=False, inplace=True)
    
    return df,cat_breeds

def get_fnames(cat_breeds, common_path = "C:/Users/leero/Projects/cat-detector/images/"):
    fnames = []

    for breed in cat_breeds:
        path_img = common_path + breed
        images = get_image_files(path_img)

        # undersampling from domestic short hair: take 1/10 of the data 
        if breed == "Domestic Short Hair":
            images = images[:5000]

        fnames += images
    return fnames


def accuracy_by_class(learn, k, classes):
    interp = ClassificationInterpretation.from_learner(learn)
    matrix = interp.confusion_matrix()

    accuracies = []
    L = len(classes)
    for i in range(L):
        accuracy = matrix[i,i]/sum(matrix[i,:])
        accuracies.append(accuracy)

    low_accuracy = np.array(accuracies)[np.argsort(accuracies)][:k]
    breeds_low_accuracy = np.array(classes)[np.argsort(accuracies)][:k]

    for i,j in zip(breeds_low_accuracy,low_accuracy):
        print(i,j)