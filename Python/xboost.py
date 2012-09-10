"""XBoost Version 0.1, 09/09/2012

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

This is the documentation for XBoost library

Copyright Xiang Zhang 2012
"""

"""!\mainpage XBoost - eXetremely fast ada-Boost library

Copyright by Xiang Zhang, 2012

"""

import numpy

class DataProc(object):
    """Data pre-processors and post-processors"""
    
    @staticmethod
    def l2mat(l, naval = None):
        """List structure to matrix conversion
        l: the list of lists
        naval: unrecognized values are replaced by naval. Default to None."""
        
        #determine value validity
        if naval != None:
            naval = float(naval)
        if len(l) == 0:
            raise ValueError("List empty")
        for row in l:
            if len(row) != len(l[0]):
                raise ValueError("List dimension inconsistent")
            
        #Convert to matrix
        mat = numpy.zeros((len(l), len(l[0])))
        for i in range(len(l)):
            for j in range(len(l[0])):
                try:
                    mat[i,j] = float(l[i][j])
                except(ValueError):
                    if naval == None:
                        raise ValueError("Non-real value encountered at row {} column {}: {} but naval is not specified".format(i,j,l[i][j]))
                    else:
                        mat[i,j] = naval
                        
        return mat
    
    @staticmethod
    def zscore(data, naval = None, replace = True):
        """Zero-out and normalize the values
        data: input data
        naval: unrecognized value indicator
        replace: whether replace every naval with the mean
        return value is a tuple (new, mu, std) storing new data, the mean and the standard deviation"""
        
        #Determine value validity
        if naval != None:
            naval = float(naval)
        
        #Iterative mean computation mu[n] = (1-1/n) mu[n-1] + 1/n val[n]
        mu = numpy.zeros(data.shape[1])
        for j in range(data.shape[1]):
            n = 1
            for i in range(data.shape[0]):
                if data[i,j] != naval:
                    mu[j] = (1-1/n)*mu[j] + 1/n*data[i,j]
                    n = n + 1
                    
        #Iterative variance computation var[n] = (1-1/n) var[n-1] + 1/n (val[n] - mu)^2
        var = numpy.zeros(data.shape[1])
        for j in range(data.shape[1]):
            n = 1
            for i in range(data.shape[0]):
                if data[i,j] != naval:
                    var[j] = (1-1/n)*var[j] + 1/n*(data[i,j] - mu[j])*(data[i,j] - mu[j])
                    n = n + 1
        std = numpy.sqrt(var)
        
        #Compute the new data
        new = numpy.zeros(data.shape)
        for j in range(data.shape[1]):
            for i in range(data.shape[0]):
                if data[i,j] != naval:
                    new[i,j] = (data[i,j] - mu[j])/std[j]
                elif replace == True:
                    new[i,j] = 0
                else:
                    new[i,j] = naval
                    
        return (new, mu, std)
    
    @staticmethod
    def rezscore(data, mu, std, naval = None, replace = True):
        """Redo zero-out and normalize using designated mu and std
        data: input data
        mu: mean vector
        std: standard deviation vector
        naval: unrecognized value indicator
        replace: whether replace every naval with the mean
        return value is a tuple (new, mu, std) storing new data, the mean and the standard deviation"""
        
        #Compute the new data
        new = numpy.zeros(data.shape)
        for j in range(data.shape[1]):
            for i in range(data.shape[0]):
                if data[i,j] != naval:
                    new[i,j] = (data[i,j] - mu[j])/std[j]
                elif replace == True:
                    new[i,j] = 0
                else:
                    new[i,j] = naval

class Classifier(object):
    """Classifier virtual object"""
    
    def __init__(self):
        """Constructor"""
        pass
    
    @staticmethod
    def preproc(labels, data):
        """Preprocessing the data
        labels: the label of the training data
        data: the data of the training data"""
        pass
    
    def learn(self, p, weights):
        """Learn from the data with weights
        p: the preprocessed data
        weights: the weights on each data item"""
        pass
    
    def infer(self, data):
        """Infer from data
        data: the testing data, should be a single vector"""
        pass
    
    def test(self, data):
        """Testing the data
        data: the testing data, should be a matrix"""
        pass
    
class StumpClassifier(Classifier):
    """Stump naive learner"""
    
    def __init__(self):
        """Constructor"""
        Classifier.__init__(self)
        self.v = None
        self.f = None
        self.c = None
        
    @staticmethod
    def preproc(labels, data, eps = 0):
        """Preprocessing the data
        labels: the label of the training data
        data: the data of the training data
        eps: the difference between values that we accept"""
        
        #sort the data
        new = numpy.zeros(data.shape)
        lab = numpy.zeros(data.shape)
        ind = numpy.argsort(data, axis = 0)
        for i in range(ind.shape[1]):
            new[:,i] = data[ind[:,i],i]
            lab[:,i] = labels[ind[:,i]]
            
        #Heuristic stump separation
        stumps = list()
        ldiff = lab[0:(lab.shape[0]-1), :] - lab[1:(lab.shape[0]),:]
        for i in range(data.shape[1]):
            lstarts = 1 + numpy.nonzero(ldiff[:,i])[0]
            lstarts = numpy.concatenate((numpy.array([0]), lstarts))
            lends = numpy.concatenate((lstarts[1:]-1, numpy.array([new.shape[0] - 1])))
            l = lab[lstarts, i]
            v = new[lstarts, i]
            vdiff = v[0:(v.shape[0]-1)] - v[1:(v.shape[0])]
            vstarts = 1 + numpy.nonzero(numpy.abs(vdiff) >= eps)[0]
            vstarts = numpy.concatenate((numpy.array([0]), vstarts))
            vends = numpy.concatenate((vstarts[1:] - 1, numpy.array([lstarts.shape[0]-1])))
            
            stump = dict()
            stump['lstarts'] = lstarts
            stump['lends'] = lends
            stump['l'] = l
            stump['vstarts'] = vstarts
            stump['vends'] = vends
            stump['v'] = v[vstarts]
            
            stumps.append(stump)
        
        #Get the return dictionary
        p = dict()
        p['ind'] = ind
        p['data'] = new
        p['label'] = lab
        p['tlabel'] = labels
        p['tdata'] = data
        p['stumps'] = stumps
        
        return p
    
    def learn(self, p, weights):
        """Learn from the data with weights
        p: the preprocessed data
        weights: the weights on each data item
        error and the classification result is returned"""
        e = float('inf')
        for i in range(p['data'].shape[1]):
            s = p['stumps'][i]
            dist = weights[p['ind'][:,i]]
            d = numpy.zeros(s['l'].shape)
            for j in range(s['l'].shape[0]):
                for k in range(s['lstarts'][j], s['lends'][j] + 1):
                    d[j] = d[j] + dist[k]
            
            pos = numpy.cumsum((s['l'] > 0) * d)
            neg = numpy.cumsum((s['l'] <= 0) * d)
            negsum = numpy.sum((s['l'] <= 0) * d)
            
            for j in range(s['v'].shape[0] - 1):
                err = pos[s['vends'][j]] + negsum - neg[s['vends'][j]]
                if err < e:
                    e = err
                    self.v = s['v'][j+1]
                    self.f = i
                    self.c = 1
                if (1-err) < e:
                    e = 1-err
                    self.v = s['v'][j+1]
                    self.f = i
                    self.c = -1
                    
        h = numpy.ones(p['tlabel'].shape)
        h[p['tdata'][:,self.f] < self.v] = -1
        if self.c == -1:
            h = -h
        return (numpy.sum((p['tlabel'] != h) * weights), h)
    
    def infer(self, data):
        """Infer from data
        data: the testing data, should be a matrix"""
        h = numpy.ones(data.shape[0])
        h[data[:,self.f] < self.v] = -1
        if self.c == -1:
            h = -h
        return h
                    
class XBoost(object):
    """XBoost object"""
    
    def __init__(self, t = 1, c = StumpClassifier):
        """Construct
        t: number of base classifier
        c: classifier base class"""
        self.t = t
        self.c = c
        self.a = numpy.zeros(self.t)
        self.m = list()
        
    def learn(self, labels, data):
        """Learn a boosing module
        labels: input labels
        data: data of labels
        Final error and classification result is returned"""
        d = numpy.ones(labels.shape[0])/labels.shape[0]
        p = self.c.preproc(labels, data)
        H = numpy.zeros(labels.shape[0])
        for t in range(self.t):
            m = self.c()
            (e,h) = m.learn(p,d)
            self.m.append(m)
            self.a[t] = 1/2*numpy.log((1-e)/e)
            
            H = H + h*self.a[t]
            z = 2*numpy.sqrt(e*(1-e))
            d = d*numpy.exp(-self.a[t]*labels*h)/z
            
        h = numpy.sign(H)
        
        return (numpy.sum(h != labels)/labels.shape[0], h)
    
    def infer(self, data, t = None):
        """Infer from data
        data: the testing data, should be a matrix
        t: the number of base classifiers used"""
        if t == None:
            t = self.t
        if t > self.t:
            raise ValueError("Number of base classifiers exceeding limit")
        
        h= numpy.zeros(data.shape[0])
        for i in range(t):
            h = h + self.a[i] * self.m[i].infer(data)
            
        return numpy.sign(h)