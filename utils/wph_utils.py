import pywph as pw
import numpy as np
import math

class WPH():
    
    def __init__(self, data, shape, J, L, dn=0,device=0,batch=False, TransfromOnStart=True):
        M, N = shape
        self._coef_name = ["S00", "S01", "S11", "C01", "Cphase"]
        self._mode_name = ["j1","theta1", "phase1", "j2", "theta2", "phase2", "n", "a", "pseudo"]
        self.J          = J
        self.L          = L
        self.dn         = dn
        self.wph_op     = pw.WPHOp(M,N,J=self.J,L=self.L,dn=self.dn, device=device)
        self.data       = data
        if batch is False and TransfromOnStart is True:
            self.doTransform()

    def getAllCoeffs(self):
        coefs   = dict()
        modes   = dict()
        for name in self._coef_name:
            coefs[name], modes[name]  = self.wph.get_coeffs(name)
        return coefs, modes
    
    def getCoef(self):
        return self.wph.get_coeffs
    
    def batchGetAllIsoCoeffs(self,batch_size=100):
        N       = self.data.shape[0]
        N_batch = math.ceil(N // batch_size)
        for i in range(N_batch):
            start = i*batch_size
            end   = min((i+1)*batch_size,N)
            self.doTransform(self.data[start:end])
            coef, _ = self.getAllIsoCoeffs()
            try:
                for key in coef.keys():
                    coefs[key] = np.r_[coefs[key], coef[key]]
            except:
                coefs = coef
        return coefs
                    
        
    def getAllIsoCoeffs(self):
        coefs   = dict()
        modes   = dict()
        for name in self._coef_name[:3]:
            coef, mode  = self.wph.get_coeffs(name)
            if self.batch == True:
                coef    = coef.reshape(self.batch_size,self.J,self.L)
                isocoef = coef.mean(axis=2)
            else:
                coef    = coef.reshape(self.J,self.L)
                isocoef = coef.mean(axis=1)
            coefs[name] = isocoef
            
        for name in self._coef_name[3:]:
            if name == "C01": 
                n = 0
                isocoef = np.zeros([self.batch_size, self.J*(self.J-1)//2], dtype=np.complex) if self.batch\
                else np.zeros([self.J*(self.J-1)//2], dtype=np.complex)
                for j2 in range(self.J):
                    for j1 in range(j2):
                        coef = np.array([self.wph.get_coeffs(name, j1=j1,j2=j2,t1=l,t2=l)[0]\
                                        for l in range(self.L)])
                        if self.batch == True:
                            isocoef[:,n] = coef.mean(axis=0).reshape(self.batch_size,)
                        else:
                            isocoef[n] = coef.mean()
                        n += 1
                            
            if name == "Cphase":
                n = 0
                isocoef = np.zeros([self.batch_size, self.J*(self.J-1)//2], dtype=np.complex) if self.batch\
                else np.zeros([self.J*(self.J-1)//2], dtype=np.complex)
                for j2 in range(self.J):
                    for j1 in range(j2):
                        coef = np.array([self.wph.get_coeffs(name, j1=j1,j2=j2,t1=l,t2=l)[0]\
                                        for l in range(self.L)])
                        if self.batch == True:
                            isocoef[:,n] = coef.mean(axis=0).reshape(self.batch_size,)
                        else:
                            isocoef[n] = coef.mean()
                        n += 1
            coefs[name] = isocoef
        
        return coefs, modes
    
    def getSphericalCoef(self, name):
        coef, mode  = self.wph.get_coeffs(name)
        return coef
 
    def doTransform(self, data=None):
        if data is None:
            data = self.data
        self.batch      = True if len(data.shape) == 3 else False
        self.batch_size = data.shape[0] if self.batch == True else None
        self.wph = self.wph_op(data,ret_wph_obj=True)
        del data

    def getCoefNames(self):
        return self._coef_name
    
    def getModeNames(self):
        return self._mode_name
    
    def getWPH(self):
        return self.wph