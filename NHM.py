# -*- coding: utf-8 -*-
import os
import random
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from numba import jit
from numpy.lib.stride_tricks import sliding_window_view as roll
from numba.typed import List

import multiprocessing


# ==============================================================================
# 0. data load & auxiliary functions
# ==============================================================================
# load north holding detail data
DATA_PATH = '/data/MultiFactorData/Data/NorthHolding/'  # north holding data path
BASE_PATH = '/data/MultiFactorData/Data/'  # database path where there are other basic data


def Load_Data(data_name, target_folder, where=None):
    """
    Load data from database.
    """
    if where:
        return pd.read_hdf(BASE_PATH + r'/%s/%s.h5' % (target_folder, data_name), data_name, where=where)
    else:
        # load without date info
        return pd.read_hdf(BASE_PATH + r'/%s/%s.h5' % (target_folder, data_name), data_name)


def LoadHoldingTensor(varName='marketValue', startDate=None, endDate=None, dataPath=DATA_PATH):
    '''
    Load holding matrices (e.g. h5 files) for each day, to be aligned to a 3D tensor indexed by (date, asset, fund)

    Params
    ---------
    startDate: start date
    endDate: end date
    dataPath: path of data files
    '''
    holdingFiles = list(sorted(os.listdir(dataPath)))
    tradeDates = Load_Data('Trade_Date_List', 'Trade_Date').set_index('date').loc[holdingFiles[0][:-3]:].index.tolist()
    holdingFiles = [f for f in holdingFiles if f[:-3] in tradeDates]
    startIdx = np.searchsorted(holdingFiles, startDate + '.h5') if startDate else 0
    endIdx = np.searchsorted(holdingFiles, endDate + '.h5') if endDate else len(holdingFiles)
    # initialize with the first date holding
    hldData = pd.read_hdf(dataPath + holdingFiles[startIdx])
    hldMtrx = hldData[varName].unstack()
    assetRows = hldMtrx.index
    agentCols = hldMtrx.columns
    holdingMtrxLst = [hldMtrx]
    dateList = [holdingFiles[startIdx].replace('.h5', '')]
    for i in range(startIdx + 1, endIdx):
        iHldData = pd.read_hdf(dataPath + holdingFiles[i])
        iHldMtrx = iHldData[varName].unstack()
        # align asset rows
        iAssetUnion = assetRows.union(iHldMtrx.index)
        if len(iAssetUnion) > len(assetRows):
            assetRows = iAssetUnion
            # print('Reshape asset to %s' % assetRows.shape[0])
            for h in range(len(holdingMtrxLst)):
                holdingMtrxLst[h] = holdingMtrxLst[h].reindex(index=assetRows)
        # align agent columns
        iAgentCols = agentCols.union(iHldMtrx.columns)
        if len(iAgentCols) > len(agentCols):
            agentCols = iAgentCols
            # print('Reshape agent to %s' % agentCols.shape[0])
            for h in range(len(holdingMtrxLst)):
                holdingMtrxLst[h] = holdingMtrxLst[h].reindex(columns=agentCols)
        # append to full data list
        iHldMtrx = iHldMtrx.reindex(index=assetRows, columns=agentCols)
        holdingMtrxLst.append(iHldMtrx)
        dateList.append(holdingFiles[i].replace('.h5', ''))
        if i % 100 == 0: print(dateList[-1])
    holdingMtrx = np.concatenate([h.fillna(0.0).values.copy('c')[None, :, :] for h in holdingMtrxLst], axis=0)
    return holdingMtrx, dateList, assetRows, agentCols


def PrepTargetReturn(adjClose='AdjVwap', where='date>="20080101"', freq=5, lag=1):
    '''
    Prepare asset return, excess of cross-section mean, and then match to intended dates
    as specified by dateDf. If dateDf is not provided, use tradedates

    Params
    ---------
    dateDf: empty data frame indexed by date list (with index name 'date')
    where: date range
    freq: look forward length
    lag: the # of days to delay
    '''
    adjClose = Load_Data(adjClose, 'Trade_Date', where)
    excessReturn = adjClose.unstack().pct_change(freq).shift(-freq - lag).dropna(how='all')
    excessReturn[:] = excessReturn.values - excessReturn.mean(axis=1).values[:, np.newaxis]
    excessReturn = excessReturn.stack()
    return excessReturn


# ==============================================================================
# 1. operators
# ==============================================================================

# ------------------------------------------------------------------------------
# Inputs:
# a. holding matrices of shape (T,A,F), e.g. holding weight, market value, volume
# b. fund/agent attributes of shape (T,F), e.g. bank-or-broker, nationality
# c. asset attributes of shape (T,A), e.g. market cap, turnover
# Intermediate outputs:
# a. holding matrix of shape (T,A,F), e.g. holding increase
# b. fund weight of shape (T,F), e.g. fund historical IR
# c. filter matrix of shape (T,F), (T,A) or (T,A,F), e.g. foreign broker
# Final outputs:
# a. asset attributes of shape (T,A)
# ------------------------------------------------------------------------------

# 0. dimension preserving operators (T,A,F) -> (T,A,F)
# 0.0 Time series dimension operation
@jit
def TSDiff(holdingMtrx, window):
    '''
    Differencing along time-series axis
    (T,A,F) -> (T,A,F)

    Params
    ---------
    holdingMtrx:
    window:
    '''
    output = np.full(holdingMtrx.shape, np.nan)
    output[window:] = holdingMtrx[window:] - holdingMtrx[:holdingMtrx.shape[0] - window]
    return output

@jit
def TSPctChg(holdingMtrx, window):
    '''
    Sign-preserving percentage change.
    (T,A,F) -> (T,A,F)

    Params
    ---------
    holdingMtrx:
    window:
    '''
    output = np.full(holdingMtrx.shape, np.nan)
    output[window:] = (holdingMtrx[window:] - holdingMtrx[:holdingMtrx.shape[0] - window]) / \
                      (np.abs(holdingMtrx[window:]) + np.abs(holdingMtrx[:holdingMtrx.shape[0] - window]) + 1e-9)
    return output

@jit
def TSShift(holdingMtrx, step):
    '''
    Time-series shift.
    (T,A,F) -> (T,A,F)

    Params
    ---------
    holdingMtrx:
    step: steps to shift forward (backward if step<0)
    '''
    output = np.full(holdingMtrx.shape, np.nan)
    if step > 0:
        output[step:] = holdingMtrx[:holdingMtrx.shape[0] - step]
    else:
        output[:step] = holdingMtrx[-step:]
    return output

@jit
def TSCorr(Mtrx1, Mtrx2, window):
    '''
    time-series corr (over window size) of Mtrx1 & Mtrx2.
    both matrixs in the form of (T,A)
    '''
    featureShape = Mtrx1.shape
    assetNum = featureShape[-1]
    retVal = np.empty(featureShape)
    for i in range(assetNum):
        for j in range(window, featureShape[0]):
            mat1 = Mtrx1[(j - window):j, i].T
            mat2 = Mtrx2[(j - window):j, i].T

            retVal[(j - 1), i] = np.corrcoef(mat1, mat2)[0][1]
    return retVal


# 0.1 Asset axis operation
def AssetNormalize(holdingMtrx, absVal=False):
    '''
    Normalize to proportion along asset axis.
    (T,A,F) -> (T,A,F)
    TODO: refactor to a general normalization function, like PanelProd

    Params
    ---------
    holdingMtrx:
    absVal:
    '''
    if absVal:
        holdingMtrx = np.abs(holdingMtrx)
    output = holdingMtrx / np.nansum(holdingMtrx, axis=1)[:, None, :]
    return output

def AssetUnitize(holdingMtrx):
    '''
    Normalize along asset axis such that w.T @ w = 1.
    (T,A,F) -> (T,A,F)
    TODO: refactor to a general normalization function, like PanelProd

    Params
    ---------
    holdingMtrx:
    '''
    holdingUnit = np.sqrt(np.nansum(holdingMtrx * holdingMtrx, axis=1))
    output = holdingMtrx / (holdingUnit[:, None, :] + 1e-9)  # add epsilon to avoid zero division
    return output


# 0.2 Fund axis operation


# 0.3 3D panel operation
@jit(forceobj=True, parallel=True)
def PanelRollSum(holdingMtrx, window, axis=0):
    '''
    rolling sum along 'axis' in holdingMtrx (in ascending order)
    default axis is the last axis.
    '''
    v = roll(holdingMtrx, window_shape=window, axis=axis).sum(axis=-1)
    orig_shape = holdingMtrx.shape
    after_shape = v.shape
    delta = orig_shape[axis] - after_shape[axis]
    Reshape = []
    for i in orig_shape:
        Reshape.append([0, 0])
    Reshape[axis][0] = delta
    v = np.pad(v, Reshape, mode='constant', constant_values=np.nan)
    return v

@jit(forceobj=True, parallel=True)
def PanelRollMean(holdingMtrx, window, axis=0):
    '''
    rolling mean along 'axis' in holdingMtrx (in ascending order)
    default axis is the last axis.
    '''
    v = roll(holdingMtrx, window_shape=window, axis=axis).mean(axis=-1)
    orig_shape = holdingMtrx.shape
    after_shape = v.shape
    delta = orig_shape[axis] - after_shape[axis]
    Reshape = []
    for i in orig_shape:
        Reshape.append([0, 0])
    Reshape[axis][0] = delta
    v = np.pad(v, Reshape, mode='constant', constant_values=np.nan)
    return v

@jit(forceobj=True, parallel=True)
def PanelRollStd(holdingMtrx, window, axis=0):
    '''
    rolling standard deviation along 'axis' in holdingMtrx (in ascending order)
    default axis is the last axis.
    '''
    v = roll(holdingMtrx, window_shape=window, axis=axis).std(axis=-1, ddof=1)
    orig_shape = holdingMtrx.shape
    after_shape = v.shape
    delta = orig_shape[axis] - after_shape[axis]
    Reshape = []
    for i in orig_shape:
        Reshape.append([0, 0])
    Reshape[axis][0] = delta
    v = np.pad(v, Reshape, mode='constant', constant_values=np.nan)
    return v

@jit(forceobj=True, parallel=True)
def PanelRollMax(holdingMtrx, window, axis=0):
    '''
    rolling maximum along 'axis' in holdingMtrx (in ascending order)
    default axis is the last axis.
    '''
    v = roll(holdingMtrx, window_shape=window, axis=axis).max(axis=-1)
    orig_shape = holdingMtrx.shape
    after_shape = v.shape
    delta = orig_shape[axis] - after_shape[axis]
    Reshape = []
    for i in orig_shape:
        Reshape.append([0, 0])
    Reshape[axis][0] = delta
    v = np.pad(v, Reshape, mode='constant', constant_values=np.nan)
    return v

@jit(forceobj=True, parallel=True)
def PanelRollMin(holdingMtrx, window, axis=0):
    '''
    rolling minimum along 'axis' in holdingMtrx (in ascending order)
    default axis is the last axis.
    '''
    v = roll(holdingMtrx, window_shape=window, axis=axis).min(axis=-1)
    orig_shape = holdingMtrx.shape
    after_shape = v.shape
    delta = orig_shape[axis] - after_shape[axis]
    Reshape = []
    for i in orig_shape:
        Reshape.append([0, 0])
    Reshape[axis][0] = delta
    v = np.pad(v, Reshape, mode='constant', constant_values=np.nan)
    return v


def PanelRollIR(holdingMtrx, window, axis=0):
    return PanelRollMean(holdingMtrx, window, axis) / PanelRollStd(holdingMtrx, window, axis)

@jit
def PanelRank(holdingMtrx, axis=-1):
    '''
    rank elements along 'axis' in holdingMtrx (in ascending order)
    default axis is the last axis.
    '''
    return holdingMtrx.argsort(axis=axis).argsort(axis=axis)

@jit
def PanelQuantile(holdingMtrx, q, axis=-1):
    '''
    return quantile value along axis. dimension is kept for compatibility.
    default axis is the last axis.
    '''

    return np.nanquantile(holdingMtrx, q=q, axis=axis, keepdims=False)

@jit
def Panelgte(holdingMtrx, N, axis=None):
    '''
    if N is a number, compare each elements in (T,A,F) with N.
    if N is an one-dimensional array, reshape (axis parameter
    denotes the axis to which the array belongs to) to make comparison along the axis.
    if greater than/equal to, True; else, False.
    '''
    holdShape = list(holdingMtrx.shape)
    panelReshape = np.ones(len(holdShape)).astype(int)
    if axis or axis == 0:
        panelReshape[axis] = N.shape[0]
    else:
        N = np.array([N])
    return np.greater_equal(holdingMtrx, N.reshape(panelReshape))

@jit
def Panellte(holdingMtrx, N, axis=None):
    '''
    if N is a number, compare each elements in (T,A,F) with N.
    if N is an one-dimensional array, reshape (axis parameter
    denotes the axis to which the array belongs to) to make comparison along the axis.
    if less than/equal to, True; else, False.
    '''
    holdShape = list(holdingMtrx.shape)
    panelReshape = np.ones(len(holdShape)).astype(int)
    if axis:
        panelReshape[axis] = N.shape[0]
    else:
        N = np.array([N])
    return np.less_equal(holdingMtrx, N.reshape(panelReshape))

@jit
def PanelProd(holdingMtrx, panelMtrx, axis=2):
    '''
    Broadcasted product (T,A,F) * (T,A) -> (T,A,F)
    Can be used for filtering effective holding.

    Params
    ---------
    holdingMtrx:
    panelMtrx:
    axis: axis along which to broadcast
    '''
    panelReshape = list(panelMtrx.shape)
    if panelReshape.__len__() < holdingMtrx.shape.__len__():
        panelReshape.insert(axis, 1)
    return holdingMtrx * panelMtrx.reshape(panelReshape)

@jit
def PanelSub(holdingMtrx, panelMtrx, axis=2):
    '''
    Broadcasted subtraction (T,A,F) * (T,A) -> (T,A,F)

    Params
    ---------
    holdingMtrx:
    panelMtrx:
    axis: axis along which to broadcast
    '''
    panelReshape = list(panelMtrx.shape)
    if panelReshape.__len__() < holdingMtrx.shape.__len__():
        panelReshape.insert(axis, 1)
    return np.subtract(holdingMtrx, panelMtrx.reshape(panelReshape))

def PanelDiv(holdingMtrx, panelMtrx, axis=2):
    '''
    Broadcasted subtraction (T,A,F) * (T,A) -> (T,A,F)

    Params
    ---------
    holdingMtrx:
    panelMtrx:
    axis: axis along which to broadcast
    '''
    panelReshape = list(panelMtrx.shape)
    if panelReshape.__len__() < holdingMtrx.shape.__len__():
        panelReshape.insert(axis, 1)
    panelMtrx = np.where(panelMtrx, panelMtrx, np.nan)
    return np.divide(holdingMtrx, panelMtrx.reshape(panelReshape))

@jit
def PanelAdd(holdingMtrx, panelMtrx, axis=2):
    '''
    Broadcasted addition (T,A,F) * (T,A) -> (T,A,F)
    Can be used for filtering effective holding.

    Params
    ---------
    holdingMtrx:
    panelMtrx:
    axis: axis along which to broadcast
    '''
    panelReshape = list(panelMtrx.shape)
    if panelReshape.__len__() < holdingMtrx.shape.__len__():
        panelReshape.insert(axis, 1)
    return np.add(holdingMtrx.astype(float), panelMtrx.reshape(panelReshape).astype(float))


# 1. dimension reduction operators
@jit
def ReducedSum(holdingMtrx, axis=2):
    return np.nansum(holdingMtrx, axis=axis)


# 2. 2D panel operators
# TODO: rolling accelerate? (do without pandas?)
@jit
def epsilon_reg(X, y, window):
    """
    do regression over 'window' length of periods, and return the residual.
    :param X: independent variable to regress on. in the form of array((T,A),(T,A),...).
        so axis 0: feature; axis 1: time; axis 2: asset.
    :param y: dependent variable of form (T,A).
    :param window: number of periods to do the regression.
    :return: residuals in the form of (T,A).
    """
    featureShape = X.shape
    assetNum = featureShape[-1]
    retVal = np.empty(y.shape)
    for i in range(assetNum):
        for j in range(window, featureShape[1]):
            indepVar = X[:, (j - window):j, i].T
            cri = np.sum(np.isnan(indepVar), axis=1) == 0
            indepVar = indepVar[cri]
            depVar = y[(j - window):j, i]
            depVar = depVar[cri]
            retV = np.linalg.lstsq(indepVar, depVar)[1]
            if len(retV) < 1:
                retVal[(j - 1), i] = np.nan  # the value remains constant. rank < number of features so returned NaN
            else:
                retVal[(j - 1), i] = retV[0]
    return retVal


def RollSum(panelMtrx, window):
    return pd.DataFrame(panelMtrx).rolling(window).sum().values


def RollMean(panelMtrx, window):
    return pd.DataFrame(panelMtrx).rolling(window).mean().values


def RollStd(panelMtrx, window):
    return pd.DataFrame(panelMtrx).rolling(window).std().values


def RollIR(panelMtrx, window):
    return RollMean(panelMtrx, window) / RollStd(panelMtrx, window)


def RollShift(panelMtrx, window):
    return pd.DataFrame(panelMtrx).shift(window).values

@jit
def TopCut(panelMtrx, q=90, axis=1):
    topPercentile = np.nanpercentile(panelMtrx, q, axis=axis)
    top = (panelMtrx > topPercentile[:, None]).astype(np.double)
    return top / np.nansum(top, axis=1)[:, None]


# 3. aggregrator
@jit
def PanelProdAgg(holdingMtrx, panelMtrx):
    '''
    Aggregrate holding by fund weight.
    (T,A,F) * (T,F) -> (T,A)

    Params
    ---------
    holdingMtrx: holding matrix of shape (T,A,F)
    panelMtrx: fund attribute
    '''
    return np.nansum(PanelProd(holdingMtrx, panelMtrx, axis=1), axis=2)

@jit
def SingleSumAgg(holdingMtrx):
    '''
    Single sum aggregrator

    Params
    ---------
    holdingMtrx: holding matrix of shape (T,A,F)
    '''
    return np.nansum(holdingMtrx, axis=2)

def AxisSum(holdingMtrx, axis):
    '''
    Aggregate holding by a specific axis.

    Params
    ---------
    holdingMtrx: holding matrix of shape (T,A,F)
    axis: along which to sum.
    '''
    ret = np.nansum(holdingMtrx, axis=axis)
    return ret / np.nansum(ret, axis=1)[:, None]


def CalcIR(holdingFactor, ar, domain, dateList, assetRows):
    # 2.2 holding factor test
    holdingFactorDf = pd.DataFrame({'factor': pd.DataFrame(holdingFactor, index=dateList, columns=assetRows).stack()})
    holdingFactorDf.index.names = ['date', 'asset']
    holdingFactorDf = holdingFactorDf.reset_index().set_index(['date', 'asset'])

    combinedData = holdingFactorDf.merge(ar, how='inner', left_index=True,
                                         right_index=True).loc[holdingFactorDf.index.levels[0][0]:]
    dailyCorr = combinedData.groupby('date').apply(lambda x: x.corr('spearman').iloc[0, 1])

    combinedDataDmn = holdingFactorDf.merge(domain, how='inner', left_index=True, right_index=True). \
                          merge(ar, how='inner', left_index=True, right_index=True).loc[
                      holdingFactorDf.index.levels[0][0]:]
    dailyCorrDmn = combinedDataDmn.groupby('date').apply(lambda x: x.corr('spearman').iloc[0, 1])
    return dailyCorr, dailyCorrDmn


# ==============================================================================
# 2. data load and factor calculation
# ==============================================================================
# load basic data as tensors
adjHolding, dateList, assetRows, agentCols = LoadHoldingTensor('adjHolding')  # marketValue
floatCap = Load_Data('Floating_Market_Cap', 'Trade_Date', where='date>="%s"' % dateList[0])
floatCap = floatCap['Floating_Market_Cap'].unstack().loc[dateList, assetRows].values.copy('c')

# prepare asset return
freq = 5
assetReturn = PrepTargetReturn(adjClose='AdjVwap', where='date>="%s"' % dateList[0], freq=freq, lag=1)
domain = Load_Data('ZZ500_Daily_Weight', 'Trade_Date', where='date>="%s"' % dateList[0]).drop(
    columns=['weight'])  # make it a pure index df

# calculate by factor expression
# f0. holding increase momentum
holdingWeight = AssetUnitize(
    PanelDiv(
        PanelRollMean(TSPctChg(adjHolding, 1), 20, axis=0),
        PanelRollStd(TSPctChg(adjHolding, 1), 20, axis=0)
    )
)
holdingFactor = AxisSum(holdingWeight, axis=2)

# primitive factor IC test
dailyCorr, dailyCorrDmn = CalcIR(holdingFactor, assetReturn, domain, dateList, assetRows)
print(
    f"""
    all IC mean:{dailyCorr.mean()}, IC std: {dailyCorr.std(ddof=1)}, IR: {dailyCorr.mean() / dailyCorr.std(ddof=1)};\n
    domain IC mean:{dailyCorrDmn.mean()}, IC std: {dailyCorrDmn.std(ddof=1)}, IR: {dailyCorrDmn.mean() / dailyCorrDmn.std(ddof=1)};\n
    """)

# raw input variables
BasicTensorVar = {}  # V1 type
BasicAssetVar = {}  # V2 type
BasicAgentVar = {}  # V3 type

# operator sets
op1Set = {}  # type 1 operators
op2Set = {}  # type 2 operators
op3Set = {}  # type 3 operators
op4Set = {}  # type 4 operators
op5Set = {}  # type 5 operators
op6Set = {}  # type 6 operators


def GenerateTensor(targetLevel, shouldLevel=None):
    '''
    Generate expression of 3D holding tensor of shape (T,A,F), with 3 routines based on type 1 - 3 operators
    '''
    if shouldLevel is None:
        targetLevel = shouldLevel
    elif shouldLevel < targetLevel:
        raise (Exception('should expand beyond target level'))
    if targetLevel == 0:
        v1Expr = random.sample(BasicTensorVar, 1)[0]
        return v1Expr, [v1Expr] + (2 ** (shouldLevel + 1) - 2) * ['VOID']
    else:
        v1Expr, v1BinTree = GenerateTensor(targetLevel - 1, shouldLevel - 1)
        rndRoutine = random.randint(1, 3)
        if rndRoutine == 1:
            # routine 1: type 1 op tensor -> tensor
            op1 = random.sample(op1Set, 1)[0]
            return f'{op1}({v1Expr})', [op1] + v1BinTree + (2 ** shouldLevel - 1) * ['VOID']
        elif rndRoutine == 2:
            # routine 2: type 2 op tensor -> tensor
            op2 = random.sample(op2Set, 1)[0]
            return f'{op2}({v1Expr})', [op2] + v1BinTree + (2 ** shouldLevel - 1) * ['VOID']
        elif rndRoutine == 3:
            # routine 3: type 3 op tensor, panel -> tensor
            op3 = random.sample(op3Set, 1)[0]
            v2Expr, v2BinTree = GenerateAssetAttr(targetLevel - 1, shouldLevel - 1)
            return f'{op3}({v1Expr},{v2Expr})', [op3] + v1BinTree + v2BinTree


def GenerateAgentAttr(targetLevel, shouldLevel=None):
    '''
    Generate agent attribute of shape (T,F), with routines based type 4 - 5 routine
    '''
    if shouldLevel is None:
        shouldLevel = targetLevel
    elif shouldLevel < targetLevel:
        raise (Exception('should expand beyond target level'))
    if targetLevel == 0:
        v3Expr = random.sample(BasicAgentVar)[0]  # raw agent attribute input
        return v3Expr, [v3Expr] + (2 ** shouldLevel - 2) * ['VOID']
    else:
        rndRoutine = random.randint(1, 2)
        if rndRoutine == 1:
            # routine 1. type 4 op, V1 -> V3
            v1Expr, v1BinTree = GenerateTensor(targetLevel - 1, shouldLevel - 1)
            op4 = random.sample(op4Set, 1)[0]
            return f'{op4}({v1Expr})', [op4] + v1BinTree + (2 ** shouldLevel - 1) * ['VOID']
        elif rndRoutine == 2:
            # routine 2. type 5 op, V3 -> V3
            v3Expr, v3BinTree = GenerateAgentAttr(targetLevel - 1, shouldLevel - 1)
            op5 = random.sample(op5Set, 1)[0]
            return f'{op5}({v3Expr})', [op5] + v3BinTree + (2 ** shouldLevel - 1) * ['VOID']


def GenerateAssetAttr(targetLevel, shouldLevel=None):
    '''
    Generate asset attribute and ALSO factor expression of shape (T,A).
    routine 1: double aggregrator(V1, V3)
    routine 2: single aggregrator(V1)
    '''
    if shouldLevel is None:
        shouldLevel = targetLevel
    elif shouldLevel < targetLevel:
        raise (Exception('should expand beyond target level'))
    if targetLevel == 0:
        v2Expr = random.sample(BasicAssetVar, 1)[0]
        return v2Expr, [v2Expr] + (2 ** shouldLevel - 2) * ['VOID']
    else:
        # generate via routines 1&2 with equal probability
        if random.random() > 0.5:
            v1Expr, v1BinTree = GenerateTensor(targetLevel - 1)  # V1
            v3Expr, v3BinTree = GenerateAgentAttr(targetLevel - 1)
            return f'PanelProdAgg({v1Expr},{v3Expr})', ['PanelProdAgg'] + v1BinTree + v3BinTree
        else:
            v1Expr, v1BinTree = GenerateTensor(targetLevel - 1)  # V1
            return f'SingleSumAgg({v1Expr})', ['SingleSumAgg'] + v1BinTree + ['NULL'] * len(v1BinTree)

# Transfer factors
# factorNames = [f for f in os.listdir(f'{BASE_PATH}Trade_Date/') if 'ori_' in f] # BASE_PATH = '/data/MultiFactorData/Data/'
# for f in factorNames:
#     fData = pd.read_hdf(f'{BASE_PATH}Trade_Date/{f}')
#     fCount = (fData!=0).sum(axis=1)
#     fData = fData.loc[fCount[fCount>100].index[0]:]
#     fData.index.name = 'date'
#     fData.to_hdf(f'/home/wenqi/MultiFactorData/Data/Express/{f}',key=f.replace('.h5',''))
#     print(f,fData.shape[0])
#
# factors = [f.replace('.h5','') for f in factorNames]
# candidateSign = pd.Series(-1.0,index=factors,dtype=np.double).to_dict()
# result = CalMaxExpoReturn(factors,candidateSign,freq=5,nbTracks=5,cost=0.0,directSave=False)
