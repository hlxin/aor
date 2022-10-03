'''
collection of chemisorption models.

newns_anderson:
'''

import torch
import numpy as np


class Chemisorption:

    def __init__(self, model_name, **kwargs):
        # Initialize the class
        if model_name == 'gnn':
            self.model_num_input = 1
        if model_name == 'newns_anderson_semi':
            self.model_num_input = 17
    
    def newns_anderson_semi(self, namodel_in, model, task , **kwargs):
        
        adse_1 = namodel_in[:,0]
        beta_1 = torch.nn.functional.softplus(namodel_in[:,1])
        delta_1 = torch.nn.functional.softplus(namodel_in[:,2])
        adse_2 = namodel_in[:,3]
        beta_2 = torch.nn.functional.softplus(namodel_in[:,4])
        delta_2 = torch.nn.functional.softplus(namodel_in[:,5])
        adse_3 = namodel_in[:,6]
        beta_3 = torch.nn.functional.softplus(namodel_in[:,7])
        delta_3 = torch.nn.functional.softplus(namodel_in[:,8])
        alpha = 0.05846361053366089
        d_cen_1 = namodel_in[:,9]
        d_cen_2 = namodel_in[:,10]
        d_cen_3 = namodel_in[:,11]
        d_cen_4 = namodel_in[:,12]
        width_1 = torch.nn.functional.softplus(namodel_in[:,13])
        width_2 = torch.nn.functional.softplus(namodel_in[:,14])
        width_3 = torch.nn.functional.softplus(namodel_in[:,15])
        width_4 = torch.nn.functional.softplus(namodel_in[:,16])
        
        idx = kwargs['batch_cif_ids']
        
        energy_DFT = self.energy[idx]
        vad2 = self.vad2[idx]
        
        if task == 'train':
            d_cen_1_DFT = self.d_cen_1[idx]
            d_cen_2_DFT = self.d_cen_2[idx]
            d_cen_3_DFT = self.d_cen_3[idx]
            d_cen_4_DFT = self.d_cen_4[idx]
            width_1_DFT = self.width_1[idx]
            width_2_DFT = self.width_2[idx]
            width_3_DFT = self.width_3[idx]
            width_4_DFT = self.width_4[idx]
            dos_ads_1_DFT = self.dos_ads_1[idx]
            dos_ads_2_DFT = self.dos_ads_2[idx]
            dos_ads_3_DFT = self.dos_ads_3[idx]
        
        ergy = self.ergy
        
        self.fermi = np.argsort(abs(ergy.detach().cpu().numpy()))[0] + 1
        
        # Semi-ellipse
        if model == 'dft':
            dos_d_1 = 1-((ergy[None,:]-d_cen_1_DFT[:,None])/width_1_DFT[:,None])**2
            dos_d_1 = abs(dos_d_1)**0.5
            dos_d_1 *= (abs(ergy[None,:]-d_cen_1_DFT[:,None]) < width_1_DFT[:,None])
            dos_d_1 += (torch.trapz(dos_d_1,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d_1 = dos_d_1 / torch.trapz(dos_d_1,ergy)[:,None]
            
            dos_d_2 = 1-((ergy[None,:]-d_cen_2_DFT[:,None])/width_2_DFT[:,None])**2
            dos_d_2 = abs(dos_d_2)**0.5
            dos_d_2 *= (abs(ergy[None,:]-d_cen_2_DFT[:,None]) < width_2_DFT[:,None])
            dos_d_2 += (torch.trapz(dos_d_2,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d_2 = dos_d_2 / torch.trapz(dos_d_2,ergy)[:,None]
            
            dos_d_3 = 1-((ergy[None,:]-d_cen_3_DFT[:,None])/width_3_DFT[:,None])**2
            dos_d_3 = abs(dos_d_3)**0.5
            dos_d_3 *= (abs(ergy[None,:]-d_cen_3_DFT[:,None]) < width_3_DFT[:,None])
            dos_d_3 += (torch.trapz(dos_d_3,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d_3 = dos_d_3 / torch.trapz(dos_d_3,ergy)[:,None]
            
            dos_d_4 = 1-((ergy[None,:]-d_cen_4_DFT[:,None])/width_4_DFT[:,None])**2
            dos_d_4 = abs(dos_d_4)**0.5
            dos_d_4 *= (abs(ergy[None,:]-d_cen_4_DFT[:,None]) < width_4_DFT[:,None])
            dos_d_4 += (torch.trapz(dos_d_4,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d_4 = dos_d_4 / torch.trapz(dos_d_4,ergy)[:,None]
            
            dos_d = (dos_d_1 + dos_d_2 + dos_d_3 + dos_d_4)/4.0
            
        else:
            dos_d_1 = 1-((ergy[None,:]-d_cen_1[:,None])/width_1[:,None])**2
            dos_d_1 = abs(dos_d_1)**0.5
            dos_d_1 *= (abs(ergy[None,:]-d_cen_1[:,None]) < width_1[:,None])
            dos_d_1 += (torch.trapz(dos_d_1,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d_1 = dos_d_1 / torch.trapz(dos_d_1,ergy)[:,None]
            
            dos_d_2 = 1-((ergy[None,:]-d_cen_2[:,None])/width_2[:,None])**2
            dos_d_2 = abs(dos_d_2)**0.5
            dos_d_2 *= (abs(ergy[None,:]-d_cen_2[:,None]) < width_2[:,None])
            dos_d_2 += (torch.trapz(dos_d_2,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d_2 = dos_d_2 / torch.trapz(dos_d_2,ergy)[:,None]
            
            dos_d_3 = 1-((ergy[None,:]-d_cen_3[:,None])/width_3[:,None])**2
            dos_d_3 = abs(dos_d_3)**0.5
            dos_d_3 *= (abs(ergy[None,:]-d_cen_3[:,None]) < width_3[:,None])
            dos_d_3 += (torch.trapz(dos_d_3,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d_3 = dos_d_3 / torch.trapz(dos_d_3,ergy)[:,None]
            
            dos_d_4 = 1-((ergy[None,:]-d_cen_4[:,None])/width_4[:,None])**2
            dos_d_4 = abs(dos_d_4)**0.5
            dos_d_4 *= (abs(ergy[None,:]-d_cen_4[:,None]) < width_4[:,None])
            dos_d_4 += (torch.trapz(dos_d_4,ergy)[:,None] <= 1e-10) / len(ergy)
            dos_d_4 = dos_d_4 / torch.trapz(dos_d_4,ergy)[:,None]
            
            dos_d = (dos_d_1 + dos_d_2 + dos_d_3 + dos_d_4)/4.0
        
        f = torch.trapz(dos_d[:,0:self.fermi],ergy[0:self.fermi])
        
        na_1, energy_NA_1, dos_ads_1 = Chemisorption.NA_Model(self, adse_1,
                                                              beta_1, delta_1,
                                                              dos_d, vad2)
        
        na_2, energy_NA_2, dos_ads_2 = Chemisorption.NA_Model(self, adse_2,
                                                              beta_2, delta_2,
                                                              dos_d, vad2)
        
        na_3, energy_NA_3, dos_ads_3 = Chemisorption.NA_Model(self, adse_3,
                                                              beta_3, delta_3,
                                                              dos_d, vad2)
        
        energy = (self.Esp
                  + (energy_NA_1 + 2*(na_1+f)*alpha*beta_1*vad2)
                  + (energy_NA_2 + 2*(na_2+f)*alpha*beta_2*vad2)
                  + (energy_NA_3 + 2*(na_3+f)*alpha*beta_3*vad2))
        
        idx = torch.from_numpy(np.array(idx, dtype=np.float32)).cuda()
        
        hybridization_energy_1 = energy_NA_1
        hybridization_energy_2 = energy_NA_2
        hybridization_energy_3 = energy_NA_3
        repulsion_energy_1 = 2*(na_1+f)*alpha*beta_1*vad2
        repulsion_energy_2 = 2*(na_2+f)*alpha*beta_2*vad2
        repulsion_energy_3 = 2*(na_3+f)*alpha*beta_3*vad2
        
        parm = torch.stack((idx,
                            energy_DFT,
                            energy,
                            d_cen_1,
                            d_cen_2,
                            d_cen_3,
                            d_cen_4,
                            width_1,
                            width_2,
                            width_3,
                            width_4,
                            adse_1,
                            beta_1,
                            delta_1,
                            adse_2,
                            beta_2,
                            delta_2,
                            adse_3,
                            beta_3,
                            delta_3,
                            hybridization_energy_1,
                            hybridization_energy_2,
                            hybridization_energy_3,
                            repulsion_energy_1,
                            repulsion_energy_2,
                            repulsion_energy_3)).T
        
        ###parm = torch.cat((parm,dos_ads_1,dos_ads_2,dos_ads_3,dos_d),dim=1)
        
        if task == 'train':
            ans = torch.cat(((energy_DFT-energy).view(-1, 1),
                             (d_cen_1_DFT-d_cen_1).view(-1, 1),
                             (d_cen_2_DFT-d_cen_2).view(-1, 1),
                             (d_cen_3_DFT-d_cen_3).view(-1, 1),
                             (d_cen_4_DFT-d_cen_4).view(-1, 1),
                             (width_1_DFT-width_1).view(-1, 1),
                             (width_2_DFT-width_2).view(-1, 1),
                             (width_3_DFT-width_3).view(-1, 1),
                             (width_4_DFT-width_4).view(-1, 1),
                             self.lamb*(dos_ads_1_DFT-dos_ads_1),
                             self.lamb*(dos_ads_2_DFT-dos_ads_2),
                             self.lamb*(dos_ads_3_DFT-dos_ads_3)),1)
            ans = ans.view(len(ans),1,-1)
        
        elif task == 'test':
            ans = energy.view(len(energy),-1)
        
        return ans, parm
    
    def NA_Model(self, adse, beta, delta, dos_d, vad2):
        h = self.h
        ergy = self.ergy
        eps = np.finfo(float).eps
        fermi = self.fermi
        
        wdos = np.pi * (beta[:,None]*vad2[:,None]*dos_d) + delta[:,None]
        wdos_ = np.pi * (0*vad2[:,None]*dos_d) + delta[:,None]
        
        # Hilbert transform
        af = torch.rfft(wdos,1,onesided=False)
        htwdos = torch.ifft(af*h[None,:,None],1)[:,:,1]
        deno = (ergy[None,:] - adse[:,None] - htwdos)
        deno = deno * (torch.abs(deno) > eps) \
               + eps * (torch.abs(deno) <= eps) * (deno >= 0) \
               - eps * (torch.abs(deno) <= eps) * (deno < 0)
        integrand = wdos / deno
        arctan = torch.atan(integrand)
        arctan = (arctan-np.pi)*(arctan > 0) + (arctan)*(arctan <= 0)
        d_hyb = 2 / np.pi * torch.trapz(arctan[:,0:fermi],ergy[None,0:fermi])
        
        lorentzian = (1/np.pi) * (delta[:,None]) \
                     / ((ergy[None,:] - adse[:,None])**2 + delta[:,None]**2)
        na = torch.trapz(lorentzian[:,0:fermi], ergy[None,0:fermi])
        
        deno_ = (ergy[None,:] - adse[:,None])
        deno_ = deno_ * (torch.abs(deno_) > eps) \
                + eps * (torch.abs(deno_) <= eps) * (deno_ >= 0) \
                - eps * (torch.abs(deno_) <= eps) * (deno_ < 0)
        integrand_ = wdos_ / deno_
        arctan_ = torch.atan(integrand_)
        arctan_ = (arctan_-np.pi)*(arctan_ > 0) + (arctan_)*(arctan_ <= 0)
        d_hyb_ = 2 / np.pi * torch.trapz(arctan_[:,0:fermi],ergy[None,0:fermi])
        
        energy_NA = d_hyb - d_hyb_
        
        dos_ads = wdos/(deno**2+wdos**2)/np.pi
        dos_ads = dos_ads/torch.trapz(dos_ads, ergy[None,:])[:,None]
        return na, energy_NA, dos_ads
    
    def gnn(self, gnnmodel_in, **kwargs):
        # Do nothing
        return gnnmodel_in, gnnmodel_in
