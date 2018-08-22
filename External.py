#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 20:10:44 2017

@author: ahmadnish
"""

import skimage.filters
import skimage.morphology
import skimage.feature

import numpy
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

import vigra


def nodeToEdgeFeat(nodeFeatures, rag):

    uv = rag.uvIds()
    uF = nodeFeatures[uv[:,0], :]
    vF = nodeFeatures[uv[:,1], :]
    feats = [ numpy.abs(uF-vF), uF + vF, uF *  vF,
             numpy.minimum(uF,vF), numpy.maximum(uF,vF)]
    return numpy.concatenate(feats, axis=1)

def computeFeatures(raw, rag):


    nrag = nifty.graph.rag

    # list of all edge features we fill 
    feats = []

    # helper function to convert 
    # node features to edge features


    # accumulate features from raw data
    fRawEdge, fRawNode = nrag.accumulateStandartFeatures(rag=rag, data=raw,
        minVal=0.0, maxVal=255.0, numberOfThreads=1)
    feats.append(fRawEdge)
    feats.append(nodeToEdgeFeat(fRawNode, rag))

    # accumulate node and edge features from
    # superpixels geometry 
    fGeoEdge = nrag.accumulateGeometricEdgeFeatures(rag=rag, numberOfThreads=1)
    feats.append(fGeoEdge)

    fGeoNode = nrag.accumulateGeometricNodeFeatures(rag=rag, numberOfThreads=1)
    feats.append(nodeToEdgeFeat(fGeoNode, rag))

    return numpy.concatenate(feats, axis=1)

import warnings
def feat_from_edge_prob(rag, raw, edge_probs, overseg):

    
    ngraph = nifty.graph
    new_feats = []

    # trivial feature
    new_feats.append(edge_probs[:,None])

    # ucm features

    edgeSizes = numpy.ones(shape=[rag.numberOfEdges])
    nodeSizes = numpy.ones(shape=[rag.numberOfNodes])

    for r in (0.01,0.1,0.2,0.4,0.5, 0.8):

        clusterPolicy = ngraph.agglo.edgeWeightedClusterPolicyWithUcm(
            graph=rag, edgeIndicators=edge_probs,
            edgeSizes=edgeSizes, nodeSizes=nodeSizes,sizeRegularizer=r)

        agglomerativeClustering = ngraph.agglo.agglomerativeClustering(clusterPolicy) 

        a_new_feat = agglomerativeClustering.runAndGetDendrogramHeight(verbose=False)
        
        new_feats.append(a_new_feat[:,None])
        

    ## begin: new features from spatial edge probabilities
    vispred = visualize(rag, overseg, edge_probs, numpy.zeros(overseg.shape))
    
    # apply several filters on visualization of the prediction (a max filter is
    # applied because Thorsten said so)     
    # sad, skimage only works on uint8 [0, 255]

    with warnings.catch_warnings():
        
        warnings.simplefilter("ignore")
        vispred = skimage.filters.rank.maximum(skimage.img_as_ubyte(vispred), 
                                           selem = numpy.ones([4,4]))
        assert(numpy.max(vispred) <= 255)
        vispred = skimage.img_as_float(vispred)
        assert(numpy.max(vispred) <= 1)
    
    imgs = []
    for sigma in [2.0, 4.0, 6.0]:
        res = nifty.filters.gaussianSmoothing(vispred, sigma)
        imgs.append(res)
        
    for sigma in [2.,4.,6.]:
        res = vigra.filters.hessianOfGaussianEigenvalues(vispred, sigma)
        numpy.save('resh',res)
        imgs.append(res[...,0])
   
    for inscale in [1.0,2.0,3.0]:
        res = vigra.filters.structureTensorEigenvalues(vispred,inscale,inscale*5.)
        imgs.append(res[...,0])
       
    nrag = nifty.graph.rag    

    for img in imgs:
        fRawEdge, fRawNode = nrag.accumulateStandartFeatures(rag=rag, data=img,
        minVal=0.0, maxVal=1.0, numberOfThreads=1)
        new_feats.append(fRawEdge)
        new_feats.append(nodeToEdgeFeat(fRawNode, rag))


    new_feats = numpy.concatenate(new_feats, axis=1)

    return new_feats

#############################################################
# Build the training set:
# ===========================
# We only use high confidence boundaries.
                
def trainingSetBuilder(featureSet, dataDset, setImages):
    
    trainingSet = {'features':[],'labels':[]}
    
    for i,z in enumerate(setImages):
        
            data = dataDset[z]
            edgeGt = data['edgeGt']    
            feats = featureSet[i]
            
            assert(feats.shape[0] == edgeGt.shape[0])
            
            where1 = numpy.where(edgeGt > 0.85)[0]
            where0 = numpy.where(edgeGt < 0.15)[0]
            
            trainingSet['features'].append(feats[where0,:])
            trainingSet['features'].append(feats[where1,:])
            trainingSet['labels'].append(numpy.zeros(len(where0)))
            trainingSet['labels'].append(numpy.ones(len(where1)))
            
    features = numpy.concatenate(trainingSet['features'], axis=0)
    labels = numpy.concatenate(trainingSet['labels'], axis=0)

    return features, labels   


def visualize(rag, overseg, edge_values, image):
    shape = overseg.shape

    for x in range(shape[0]):
        for y in range(shape[1]):

            lu = overseg[x,y]

            if x + 1 < shape[0]:
                lv = overseg[x+1,y]

                if lu != lv :
                    e = rag.findEdge(lu, lv)

                    # normalization?
                    image[x,y]   = edge_values[e]
                    image[x+1,y] = edge_values[e]


            if y + 1 < shape[1]:

                lv = overseg[x,y+1]

                if lu != lv :
                    e = rag.findEdge(lu, lv)

                    # normalization?
                    image[x,y]   = edge_values[e]
                    image[x,y+1] = edge_values[e]

    return image


