from glob import glob
import mahotas
import os
import numpy as np
import sklearn
import milk
import pylab as p


def features_haralick(img):
    return mahotas.features.haralick(img).mean(0)


def feature_lbp(img):
    return mahotas.features.lbp(img, 1, 4)

def feature_ftas(img):
    return mahotas.features.pftas(img)



# Compute Haralick texture features
# mahotas.features.haralick(f, ignore_zeros=False, preserve_haralick_bug=False, compute_14th_feature=False, return_mean=False, return_mean_ptp=False, use_x_minus_y_variance=False, distance=1)
#
# Compute Linear Binary Patterns
# mahotas.features.lbp(image, radius, points, ignore_zeros=False)
#
# Compute parameter free Threshold Adjacency Statistics
# mahotas.features.pftas(img, T={mahotas.threshold.otsu(img)})

# Compute Threshold Adjacency Statistics
# mahotas.features.tas(img)

# mahotas.features.zernike_moments(im, radius, degree=8, cm={center_of_mass(im)})


# fafterpca = milk.pca(feature)


def fileNames():
    imgFiles = open('imgFileDirs.txt', 'w+')
    for root, dirs, files in os.walk('E:\MLdata\data'):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                dna = os.path.splitext(file)[0] + '_dna.png'
                pro = os.path.splitext(file)[0] + '_pro.png'
                position = os.path.splitext(file)[0] +'.txt'
                print(os.path.join(root, dna) + " " + os.path.join(root, pro) + " \n")
                imgFiles.write(os.path.join(root, dna) + " " + os.path.join(root, pro) + " " + os.path.join(root, position) + " \n")
    imgFiles.close()


def main():
    # fileNames()
    imgFileName = open('imgFileDirs.txt', 'r')

    cnt = 0
    for line in imgFileName.readlines():
        cnt += 1
        if cnt <= 0:
            continue

        line = line.strip('\n').split()
        dnaDir = line[0]
        proDir = line[1]
        posDir = line[2]
        code = dnaDir.split('\\')[3].split('_')[1]

        featureFile = open('features/' + code + '.txt', 'a+')
        proimg = mahotas.imread(proDir)
        with open(posDir) as posFile:
            for line in posFile.readlines():
                line = line.strip('\n').split()
                x = int(line[0])
                y = int(line[1])
                if x <= 50 or x >= 2950 or y <= 50 or y >= 2950:
                    continue
                curPatch = proimg[x-49:x+50, y-49:y+50]
                patchFeatureHaralick = features_haralick(curPatch)
                print(patchFeatureHaralick)

                patchFeatureLBP = feature_lbp(curPatch)
                patchProFeature = np.hstack((patchFeatureHaralick, patchFeatureLBP))

                for num in patchProFeature:
                    featureFile.write(str(num) + " ")
                featureFile.write(' \n')
        featureFile.close()
        print(cnt)

    imgFileName.close()


def transform():
    cnt = 0
    for root, dirs, files in os.walk('E:\MLdata\HPA_ieee_test_new\\testdata'):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                featureFileDir = os.path.splitext(file)[0] + '_feature.txt'
                featureFileDir = str(os.path.join(root, featureFileDir))
                code = featureFileDir.split('\\')[4]

                featureFile = open('testFeature/' + str(code) + '.txt', 'a+')
                featureFile.write(open(featureFileDir).read() + ' \n')
                featureFile.close()
                cnt += 1
                print(cnt)


if __name__ == '__main__':
    # fileNames()
    # main()
    transform()
