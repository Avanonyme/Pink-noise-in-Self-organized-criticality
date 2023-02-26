"""
2D Abelian Sandpile

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf

class Continuous_dissipative_sandpile:

    def __init__(self,N,nb_sim,nb_gen) -> None:
        print("INIT...",end="")
        #INIT LATTICE
        self.rng=np.random.default_rng()#flat distribution for adding slope

        self.size=N #L=NxN
        self.lattice=np.zeros((self.size+1,self.size+1)) #init_lattice
        self.crit_E=1 #critical point
        self.dissipation=0.01 #dissipation
        self.coordination=4

        #ITERATIONS AND Burning
        self.sim=nb_sim
        self.gen=nb_gen
        self.burn=self.gen//4

        #INIT QUANTITIES
        self.total_grain=np.array([])
        self.outflux=np.array([])
        self.corr=np.array([])

        print("Done.")
    
    ###MAIN###
    def start_sim(self):
        """
        start sim with n gen, calculate required observables
        """
        #INIT MULTI-SIM OBSERVABLES
        self.avalanche_size=np.zeros((self.lattice.size*10))
        self.space_corr_MEAN=np.zeros((self.size))
        self.total_grain_MEAN=np.zeros((self.gen+1))
        self.outflux_MEAN=np.zeros((self.gen+1))
        self.corr_MEAN=np.zeros((self.gen+1))
        
        #Loop of m simulations
        for m in range(self.sim):
            print(f"sim n.{m}")
            self.__init__(self.size,self.sim,self.gen)

            #MAIN LOOP
            for n in range(self.gen+1):
                self.count=n
                coord=(self.rng.integers(0,self.size),1)

                self.add_energy(coord)
                self.update_gen()
                print(self.count,"\r",end="")

            self.update_sim()
        
        #NORMALISE
        self.total_grain_MEAN=self.total_grain_MEAN/(self.sim*np.max(self.total_grain_MEAN))
        self.corr_MEAN=self.corr_MEAN/(self.sim*np.max(self.corr_MEAN))
        self.outflux_MEAN=self.outflux_MEAN/(self.sim*np.max(self.outflux_MEAN))
        
        #PSD
        self.ps_grain,self.freqs_grain=self.PSD(self.total_grain_MEAN)
        self.ps_outflux,self.freqs_outflux=self.PSD(self.outflux_MEAN)

        #PLOT
        self.plot()

        return self.freqs_grain[int((self.gen+1)*(4/10)):],self.ps_grain[int((self.gen+1)*(4/10)):],np.arange(100,self.avalanche_size.size),self.avalanche_size[100:]
    
    ###UPDATES###
    def update_gen(self):

        #check for critical heights
        if np.max(self.lattice)>=self.crit_E:
            #topples critical point in while loop
            self.toppling()
            #update quantities
            self.outflux=np.append(self.outflux,self.outflux_micro)
            self.avalanche_size[np.clip(self.avalanche_size_micro,0,self.lattice.size*10-1)]+=1

        else: #if no critical points, update quantities
            self.outflux=np.append(self.outflux,0)
            self.avalanche_size[0]+=1

        self.total_grain=np.append(self.total_grain, np.sum(self.lattice))
        self.corr=np.append(self.corr,self.lattice[3,10]+self.lattice[-4,15])
    def update_sim(self):

        self.total_grain_MEAN+=self.total_grain
        self.corr_MEAN+=self.corr

    def neighbors_energy(self,coord):
        try:
            self.lattice[1:,:][coord[:-1,:]] += self.lattice[coord]/self.coordination*(1-self.dissipation) #up
            self.lattice[:-1,:][coord[1:,:]] += self.lattice[coord]/self.coordination*(1-self.dissipation) #down
            self.lattice[:,1:][coord[:,:-1]] += self.lattice[coord]/self.coordination*(1-self.dissipation) #left
            self.lattice[:,:-1][coord[:,1:]] += self.lattice[coord]/self.coordination*(1-self.dissipation) #right

        #Handle unresolved errors at down boundary
        except:
            self.lattice[-1,1:]=0
            pass

    def add_energy(self,coord):
        eps=self.rng.uniform(0,100)
        self.lattice[coord]+=eps

    def toppling(self):
        self.outflux_micro=0
        self.avalanche_size_micro=0

        while np.max(self.lattice)>=self.crit_E:
            n_crit= self.lattice>=self.crit_E

            self.neighbors_energy(n_crit)
            self.lattice[n_crit]=0

            #Add boundary dissipation
            self.outflux_micro+=np.sum(self.lattice[0:,-1])

            #closed boundary on the left-x side and both y sides
            self.lattice[0:,1]+=self.lattice[0:,0]
            self.lattice[0:,0]=0
            
            self.lattice[1,:]+=self.lattice[0,:]
            self.lattice[0,:]=0

            self.lattice[-2,:]+=self.lattice[-1,:]
            self.lattice[-1,:]=0

            #reset boundaries
            self.lattice[0:,-1]=0


            self.avalanche_size_micro+=np.where(n_crit==True)[0].size


     #COMPUTE QUANTITIES

    def PSD(self,observable):
        ps= np.abs(np.fft.fft(observable[self.burn:]))[1:]**2
        freqs= np.fft.fftfreq(observable[self.burn:].size)[1:]
        idx=np.argsort(freqs)

        return ps[idx],freqs[idx]

    ###PLOT###
    def plot(self):

        path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"Ds_sandpile")
        try: os.mkdir(path)
        except: pass
        path=path+"/"

        #plot trivial quantities
        plt.title("État final de la pile de sable")
        plt.xlabel("X");plt.ylabel("Y")
        plt.imshow(self.lattice,vmin=0,vmax=10)
        plt.savefig(path+"lattice.png")
        plt.close()
        
        plt.title("Flux dissipée de grain de sable en fonction du temps")
        plt.xlabel("Temps (t)"); plt.ylabel("Nb. grain de sable")
        plt.plot(np.arange(self.burn,self.gen+1),self.outflux_MEAN[self.burn:])
        plt.savefig(path+"outflux.png")
        plt.close()


        #compute required quantities

        #avalanche size
        plt.title("Distribution de la taille des avalanches")
        plt.xlabel("Taille des avalanches (s)"); plt.ylabel("P(s)")
        plt.loglog(np.arange(self.lattice.size*10),self.avalanche_size)
        plt.savefig(path+"avalanche_size")
        plt.close()

        #total grain
        plt.title("Puissance spectrale du nombre de grain de sable\n en fonction des fréquences")
        plt.xlabel("Fréquences (f) (Hz)"); plt.ylabel("S(f) (W^2/Hz)")
        plt.loglog(self.freqs_grain,self.ps_grain,label="Données")
        plt.loglog(self.freqs_grain,1/self.freqs_grain,label="Pente 1/f")
        plt.legend()
        plt.savefig(path+"total_grain")
        plt.close()

        plt.title("Puissance spectrale du flux extérieur\n en fonction des fréquences")
        plt.xlabel("Fréquences (f) (Hz)"); plt.ylabel("S(f) (W^2/Hz)")
        plt.loglog(self.freqs_outflux[int((self.gen+1)*(4/10)):],self.ps_outflux[int((self.gen+1)*(4/10)):],label="Données")
        plt.loglog(self.freqs_outflux[int((self.gen+1)*(4/10)):],1/self.freqs_outflux[int((self.gen+1)*(4/10)):],label='Pente 1/f')
        plt.legend()
        plt.savefig(path+"outflux_PSD")
        plt.close()

        #Autocorrelation (validation)

        plot_acf(self.corr_MEAN[:-10000],lags=np.arange(self.corr_MEAN[:-10000].size))
        plt.title("Auto-corrélation temporelle \n du nombre de grains par site")
        plt.xlabel("Période (t)"); plt.ylabel("Auto-corrélation C(t)")
        plt.savefig(path+"auto_correlation")
        plt.close()

if __name__=="__main__":
    N=50
    nb_sim=1
    nb_gen=30000
    Sim=Continuous_dissipative_sandpile(N,nb_sim,nb_gen)
    freq_grain,ps_grain,avalanche_range,avalanche_dist=Sim.start_sim()
   

    from scipy.optimize import curve_fit
    inverse_avalanche_range=1/avalanche_range
    avalanche_slope,avalanche_cov=curve_fit(lambda x,a,b: a*x**b,inverse_avalanche_range,avalanche_dist,p0=(avalanche_dist[0],1))

    inverse_freqs_grain=1/freq_grain
    total_grain_slope,total_grain_cov=curve_fit(lambda x,a,b:a*x**b,inverse_freqs_grain,ps_grain,p0=(ps_grain[0],1))
    print(avalanche_slope,total_grain_slope)


    plt.title("Distribution de la taille des avalanches\n (Ds)")
    plt.loglog(avalanche_range,avalanche_dist,label="Données")
    plt.plot(avalanche_range,avalanche_slope[0]/avalanche_range**avalanche_slope[1],label=f'a={avalanche_slope[1]}±{np.sqrt(np.diag(avalanche_cov))[1]}')
    plt.legend()
    plt.savefig('Ds_sandpile/avalanche_size_fit')
    plt.close()

    plt.title("Puissance spectrale du nombre de grain de sable\n en fonction des fréquences\n (DS)")
    plt.xlabel("Fréquences (f) (Hz)"); plt.ylabel("S(f) (W^2/Hz)")
    plt.loglog(freq_grain,ps_grain,label="Données")
    plt.plot(freq_grain,total_grain_slope[0]/freq_grain**total_grain_slope[1],label=f'a={total_grain_slope[1]}±{np.sqrt(np.diag(total_grain_cov))[1]}')
    plt.legend()
    plt.savefig("Ds_sandpile/total_grain_fit")
    plt.close()
        

