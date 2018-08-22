import numpy
import scipy
import pylab
import os
import urllib.request
import zipfile
import skimage.io
import skimage.filters
import skimage.morphology
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('/Users/ahmadnish/dev/bld/nifty/python')
import nifty
import nifty.graph
import nifty.graph.agglo
import nifty.segmentation
import nifty.filters
import nifty.graph.rag
import nifty.ground_truth
import nifty.graph.opt.multicut
from External import *

from random import randint


from pylab import imshow
from skimage.morphology import h_minima
from nifty.filters import gaussianSmoothing


import vigra
#############################################################
# Setup Datasets:
# ===============

# number of images taken from the database              
nImg = int(30)
plot = 'on'

# Setting number if sets for Autocontex to be applied
AutocontextDepth = int(4)
random = 'on'

# Random Image Collections (used for high depth Autocontext (>6))
if AutocontextDepth > 8 or random == 'on':
    random_set = 'on'
else:
    random_set = 'off'

# leaves one fifth of the train database for benchmarking
devider = int(nImg - nImg/5)

# Initializing the images sets
setImages = []
# Initializing Autocontext 
if (AutocontextDepth):
    # calculates number of images per set
    if(random_set == 'on'):
        ips = int(6) 
    else:
        ips = int(devider / AutocontextDepth)
    
    for i in range(AutocontextDepth):
        img = []
        for j in range(ips):
            if(random_set == 'on'):
                img.append(randint(0,devider-1))
            else:
                img.append(i*ips + j)
        setImages.append(img)

else:
    setImages.append(list(range(devider)))
    AutocontextDepth = int(1)
    ips = devider

# load ISBI 2012 raw and probabilities
# for train and test set
# and the ground-truth for the train set

rawDsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/raw_train.tif')[0:devider, ...],
    'test' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/raw_train.tif')[devider:nImg, ...], #27 should become devider
}

gtDsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/groundtruth.tif')[0:devider, ...],
    'test'  : skimage.io.imread('NaturePaperDataUpl/ISBI2012/groundtruth.tif')[devider:nImg, ...], #27 should become devider
}

computedData = {
    'train' : [{} for z in range(rawDsets['train'].shape[0])],
    'test'  : [{} for z in range(rawDsets['test'].shape[0])]
}


#############################################################
#  Over-segmentation, RAG & Extract Features:
#  ============================================

print("Precalculations(overseg, RAG, Image Features Extraction) on the Database ...")

for ds in ['train','test']:
    
    rawDset = rawDsets[ds]
    gtDset = gtDsets[ds]
    dataDset = computedData[ds]


    for z in range(rawDset.shape[0]):   #
        print(".....image number {} dataset {}".format(z+1, ds))
        data = dataDset[z]

        raw  = rawDset[z, ... ]

        # oversementation
        fraw = vigra.filters.hessianOfGaussianEigenvalues(raw.astype('float32')/255, 2)
        # select first eigenvalue
        fraw = fraw[...,0]
  
      
        data['fraw'] = fraw

        overseg = nifty.segmentation.seededWatersheds(fraw, method='node_weighted')
        overseg -= 1
        data['overseg'] = overseg
        
        rag = nifty.graph.rag.gridRag(overseg)
        data['rag'] = rag

        features = computeFeatures(raw=fraw, rag=rag)
        
        # print(features.shape)

        data['features'] = features

        gtImage = gtDset[z, ...] 
        
        seeds = nifty.segmentation.localMaximaSeeds(gtImage)        

        growMap = nifty.filters.gaussianSmoothing(1.0-gtImage, 1.0)
        growMap += 0.1*nifty.filters.gaussianSmoothing(1.0-gtImage, 6.0)
        gt = nifty.segmentation.seededWatersheds(growMap, seeds=seeds)

        # for benchmarking purposes...
        if(ds == 'test'):
            data['seeds'] = seeds
            data['gt'] = gt

        overlap = nifty.ground_truth.overlap(segmentation=overseg, 
                                   groundTruth=gt)

        edgeGt = overlap.differentOverlaps(rag.uvIds())
        data['edgeGt'] = edgeGt

        assert(rag.numberOfEdges == features.shape[0])
        assert(edgeGt.shape[0] == features.shape[0])

        # plot an image from each set
        if z % 12 == 0 and plot == 'on' :
            figure = pylab.figure()
            figure.suptitle('%sing Set Slice %d'%(ds,z), fontsize=16)

            #fig = matplotlib.pyplot.gcf()
            figure.set_size_inches(18.5, 10.5)

            figure.add_subplot(3, 2, 1)
            pylab.imshow(raw, cmap='gray')
            pylab.title("Raw data %s"%(ds))

            figure.add_subplot(3, 2, 2)
            pylab.imshow(fraw, cmap='gray')
            pylab.title("Filtered Raw data %s"%(ds))

            figure.add_subplot(3, 2, 3)
            pylab.imshow(nifty.segmentation.segmentOverlay(raw, overseg, 0.2, thin=False))
            pylab.title("Superpixels %s"%(ds))

            figure.add_subplot(3, 2, 4)
            pylab.imshow(seeds, cmap=nifty.segmentation.randomColormap(zeroToZero=True))
            pylab.title("Partial ground truth %s" %(ds))

            figure.add_subplot(3, 2, 5)
            pylab.imshow(nifty.segmentation.segmentOverlay(raw, gt, 0.2, thin=False))
            pylab.title("Dense ground truth %s" %(ds))
            pylab.tight_layout()
            pylab.savefig('output/Precalculations_%sImg_number %d.pdf'%(ds, z))


#############################################################
# Train the random forests (RF):
# ===============================

# Applies Multicut on every prediction from RF
MulticutTraining = 'off' 

# Uses only Context features for training the new RFs
onlyContext = 'off'

ds = 'train'

gtDset = gtDsets[ds]; 
dataDset = computedData[ds]

rf = [] # List for storing the Random Forest Classifiers
ff = [] # Has nothing to do with the code, only informative
for i in range(AutocontextDepth): # Looping over image Sets
    setFeatures = []
    for z in setImages[i]: #looping over every image in set
        
        data = dataDset[z]
        fraw = data['fraw']
        rag = data['rag']
        features = data['features']
        gtImage = gtDset[z, ...]
        overseg = data['overseg']

        if(i is 0):
            setFeatures.append(features)
        else:
            for j in range(i):
                predictions = rf[j].predict_proba(features)[:,1]

                if(MulticutTraining == 'on'):

                    MulticutObjective = rag.MulticutObjective

                    eps =  0.00001
                    p1 = numpy.clip(predictions, eps, 1.0 - eps) 
                    weights = numpy.log((1.0-p1)/p1)

                    objective = MulticutObjective(rag, weights)
                    solver = MulticutObjective.greedyAdditiveFactory().create(objective)
                    arg = solver.optimize(visitor=MulticutObjective.verboseVisitor())
                    result = nifty.graph.rag.projectScalarNodeDataToPixels(rag, arg)
                    growMap = nifty.filters.gaussianSmoothing(1.0-gtImage, 1.0)
                    growMap += 0.1*nifty.filters.gaussianSmoothing(1.0-gtImage, 6.0)
                    gt = nifty.segmentation.seededWatersheds(growMap, seeds=result)

                    overlap = nifty.ground_truth.overlap(segmentation=overseg, 
                                           groundTruth=gt)

                    predictions = overlap.differentOverlaps(rag.uvIds())
                    
                new_f = feat_from_edge_prob(rag, fraw, predictions, overseg)
                # print(new_f.shape)

                if(onlyContext == 'on'):
                    features = new_f
                else:
                    features = numpy.concatenate((data['features'],new_f),axis = 1)

            setFeatures.append(features)
    
    
    ff.append(numpy.concatenate(setFeatures, axis = 0)) #Informative
    features, labels = trainingSetBuilder(setFeatures, dataDset, setImages[i])
    print(ff[i].shape,features.shape,labels.shape)
    rf.append(RandomForestClassifier(n_estimators=200, oob_score=True))
    print('training Random Forest {} '.format(i+1))
    rf[-1].fit(features, labels)
    print("OOB SCORE",rf[-1].oob_score_)


#############################################################'
# Predict Edge Probabilities & Optimize Multicut Objective:
# ===========================================================
# Now we gonna use the test images on the random forests we made
# The way we apply this is the same way we trained the Random Forests
# That is we run every image iteratively over all the random forests

ds = 'test'
    
rawDset = rawDsets[ds]
gtDset = gtDsets[ds]
dataDset = computedData[ds]


p = []
end_result= []

for z in range(rawDset.shape[0]):   
    
    data = dataDset[z]
    raw = rawDset[z,...]
    fraw = data['fraw']
    overseg = data['overseg']
    rag = data['rag']
    edgeGt = data['edgeGt']    
    features = data['features']
    gt = data['gt']
    seeds = data['seeds']

    args = []
    results = []
    for i in range(AutocontextDepth):
        predictions = rf[i].predict_proba(features)[:,1]
        
        # setup multicut objective
        MulticutObjective = rag.MulticutObjective   
       
        eps =  0.00001
        p1 = numpy.clip(predictions, eps, 1.0 - eps) 
        weights = numpy.log((1.0-p1)/p1)
    
        objective = MulticutObjective(rag, weights)
        solver = MulticutObjective.greedyAdditiveFactory().create(objective)
    
        arg = solver.optimize(visitor=MulticutObjective.verboseVisitor())
        result = nifty.graph.rag.projectScalarNodeDataToPixels(rag, arg)
        
        new_f = feat_from_edge_prob(rag, fraw, predictions, overseg)
        
        if(onlyContext == 'on'):
            features = new_f
        else:
            features = numpy.concatenate((data['features'],new_f),axis = 1)
        
        args.append(arg)
        results.append(result)
    
    end_result.append(args)

    p.append(predictions)



    ## plot all the test set
    if plot == 'on':
        figure = pylab.figure()        

        figure.set_size_inches(18.5, 10.5)

        figure.add_subplot(2, 3, 1)
        pylab.imshow(results[0], cmap=nifty.segmentation.randomColormap())
        pylab.title("1st RF restult")

        figure.add_subplot(2, 3, 2)
        pylab.imshow(result, cmap=nifty.segmentation.randomColormap())
        pylab.title("Deepest RF results")

        figure.add_subplot(2, 3, 3)
        pylab.imshow(seeds, cmap=nifty.segmentation.randomColormap(zeroToZero=True))
        pylab.title("Ground Truth")

        figure.add_subplot(2, 3, 4)
        pylab.imshow(nifty.segmentation.segmentOverlay(raw, results[0], 0.2, thin=False))
        pylab.title("1st RF restult")

        figure.add_subplot(2, 3, 5)
        pylab.imshow(nifty.segmentation.segmentOverlay(raw, result, 0.2, thin=False))
        pylab.title("latest RF result")

        figure.add_subplot(2, 3, 6)
        pylab.imshow(nifty.segmentation.segmentOverlay(raw, gt, 0.2, thin=False))
        pylab.title("Ground Truth")
        pylab.tight_layout()
        pylab.savefig('output/testing_testRF%d_Img%d.pdf' %(i, z))

### copied from auto5 for benchmarking and comparing purposes
# dimensions: test_images, random_forests, benchmarks
error = numpy.zeros([rawDset.shape[0], AutocontextDepth, 2])
# for each slice
for z in range(rawDset.shape[0]):
        
    data = dataDset[z]
    seeds = data['seeds']
    rag = data['rag']
    multicutResults_per_img = end_result[z]
    for r in range(AutocontextDepth):
        
        multicutResults_per_rf = multicutResults_per_img[r]        
        seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, multicutResults_per_rf)    
        randError = nifty.ground_truth.RandError(seeds, seg, ignoreDefaultLabel = True)
        variationError = nifty.ground_truth.VariationOfInformation(seeds, seg, ignoreDefaultLabel = True)
        error[z,r,:] = [randError.error, variationError.value]

mean_error = numpy.mean(error, axis = 0)
print(mean_error)
###################
"""

dataDset = computedData['test']
for i in range(rawDset.shape[0]):
    data = dataDset[i]
    edgeGt = data['edgeGt']
    assert(edgeGt.shape[0] == p[i].shape[0])
    # randError = nifty.ground_truth.RandError(edgeGt, p[i], ignoreDefaultLabel = True)
    obj = nifty.ground_truth.VariationOfInformation(edgeGt, p[i],ignoreDefaultLabel = True)
    print('variation information score for image {} is: {}'.format(i,obj.value))
"""