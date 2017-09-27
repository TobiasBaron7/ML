from PIL import Image
import os
import time

#location of folders with pgm images in each subfolder
imageSourcePath = '../data/yaleB/ExtendedYaleB/'
#destination path, where to save converted images
imageDestPath   = 'data/yaleB/ExtendedYaleB_jpg'

#L is 8bit grayscale
#others are RGB(3x8bit), CMYK (4x8bit),...
_mode = 'L'


#walker = os.walk(imageSourcePath) - alternative way of getting subdirectories

totalImages = 0
for folder in os.listdir(imageSourcePath):
    for img in os.listdir(imageSourcePath + '/' + folder):
        if img.endswith('.info'):
            continue
        else:
            totalImages += 1



#iterate over each folder, which are found at imageSourcePath
#iterate over each file in current folder
#ignore non-image files and convert images (like RGB) to _mode
#save image to destination path
#counts processed images
counter = 0
#error counter
errors = 0
for folder in os.listdir(imageSourcePath):
    currPath = imageSourcePath + '/' + folder
    destPath = imageDestPath + '/' + folder
    for img in os.listdir(currPath):
        #ignore .info files, which are no images but text
        if img.endswith('.info'):
            continue
        else:
            print(counter, '/', totalImages, end='\r')
            counter += 1
            try:
                #create folder at destination path if not existent yet
                if not os.path.exists(destPath):
                    os.makedirs(destPath)
                im = Image.open(currPath + '/' + img)
                #convert image to _mode if current mode is different
                #otherwise saving can produce errors
                if im.mode is not _mode:
                    im = im.convert(mode=_mode)
                im.save(destPath + '/' + img.replace('.pgm', '.jpg'))
            except Exception as e:
                print('Exception:\n', e)
                errors += 1


print('Processed', counter, '/', totalImages, 'images')
print('Errors:', errors)
