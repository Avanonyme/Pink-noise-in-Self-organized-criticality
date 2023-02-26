import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf


class Abelian_sandpile:

    def __init__(self,N,nb_sim,nb_gen) -> None:
        print("INIT...",end="")
        #INIT LATTICE
        self.rng=np.random.default_rng()#flat distribution for adding slope

        self.size=N #L=NxN
        self.lattice=np.zeros((self.size+1,self.size+1)) #init_lattice
        self.crit_height=4 #cricical point

        #ITERATIONS AND Burning
        self.sim=nb_sim
        self.gen=nb_gen
        self.burn=self.gen//4

        #INIT QUANTITIES
        self.total_grain=np.array([])
        self.outflux=np.array([])
        self.corr=np.array([])
        print("Done.")

    def start_sim(self):
        """
        start sim with n gen, calculate required observables
        """        
        #INIT MULTI-SIM OBSERVABLES
        self.avalanche_size=np.zeros((self.lattice.size*10))
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
                coord=self.rng.integers(0,self.size,2) #random coord    #coord=[self.size//2,self.size//2]

                #add slope unit and update quantities
                self.add_slope(coord)
                self.update_gen()
                print(self.count,end="\r")
            #update on all sim
            self.update_sim()

        #NORMALISE
        self.total_grain_MEAN=self.total_grain_MEAN/(self.sim*np.max(self.total_grain_MEAN))
        self.outflux_MEAN=self.outflux_MEAN/(self.sim*np.max(self.outflux_MEAN))
        self.corr_MEAN=self.corr_MEAN/(self.sim*np.max(self.corr_MEAN))
        
        #PSD
        self.ps_grain,self.freqs_grain=self.PSD(self.total_grain_MEAN)
        self.ps_outflux,self.freqs_outflux=self.PSD(self.outflux_MEAN)
        

        self.plot()

        return self.freqs_grain[int((self.gen+1)*(4/10)):],self.ps_grain[int((self.gen+1)*(4/10)):],np.arange(100,self.avalanche_size.size),self.avalanche_size[100:]
    def update_gen(self):

        #check for critical heights
        if np.max(self.lattice)>=self.crit_height:
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
        self.outflux_MEAN+=self.outflux
        self.corr_MEAN+=self.corr
    
    def toppling(self):
        #observables temporaires
        self.outflux_micro=0
        self.avalanche_size_micro=0

        while np.max(self.lattice)>=self.crit_height: #this loop is 1 unit of micro-time
             crit_slope= self.lattice>=self.crit_height
                
             self.lattice-=self.toppling_matrix(crit_slope) #with crit_slope the coords with height>=crit_height

            #sum on boundaries
             self.outflux_micro+=np.sum(self.lattice[0,:])+np.sum(self.lattice[:,0])+np.sum(self.lattice[-1,:])+np.sum(self.lattice[:,-1])
            #open boundaries
             self.lattice[0,:] = 0
             self.lattice[self.size,:] = 0
             self.lattice[:,0] = 0
             self.lattice[:,self.size] = 0

             self.avalanche_size_micro+=np.where(crit_slope==True)[0].size

    def toppling_matrix(self,coords):
        """
        adjacency matrix of the lattice at coord(s) (i1,j1), (i2,j2)...(im,jm)
        m=#{site with height>=crit_height}
        """
        adj_matrix=np.zeros((self.size+1,self.size+1))

        # adj_matrix_nn=4
        adj_matrix[coords] += self.crit_height
        
        # adj_matrix_n'n=-1 for n' nearest neighbors
        adj_matrix[1:,:][coords[:-1,:]] -= self.crit_height/4
        adj_matrix[:-1,:][coords[1:,:]] -= self.crit_height/4
        adj_matrix[:,1:][coords[:,:-1]] -= self.crit_height/4
        adj_matrix[:,:-1][coords[:,1:]] -= self.crit_height/4

        return adj_matrix

    def add_slope(self,coord):
        #add slope units
        self.lattice[coord[0],coord[1]]+=1
        

    ###COMPUTE QUANTITIES###        
    def PSD(self,observable):
        #code taken from
        #https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python
        ps= np.abs(np.fft.fft(observable[self.burn:]))[1:]**2
        freqs= np.fft.fftfreq(observable[self.burn:].size)[1:]
        idx=np.argsort(freqs)

        return ps[idx],freqs[idx]

    ###PLOT###    
    def plot(self):

        path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"As_sandpile")
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

        #avalanche size
        plt.title("Distribution de la taille des avalanches")
        plt.xlabel("Taille des avalanches (s)"); plt.ylabel("P(s)")
        plt.loglog(np.arange(self.lattice.size*10),self.avalanche_size)
        plt.savefig(path+"avalanche_size")
        plt.close()

        #computed quantities

        #total grain
        plt.title("Puissance spectrale du nombre de grain de sable\n en fonction des fréquences")
        plt.xlabel("Fréquences (f) (Hz)"); plt.ylabel("S(f) (W^2/Hz)")
        plt.loglog(self.freqs_grain,self.ps_grain,label="Données")
        plt.loglog(self.freqs_grain[1:],1/self.freqs_grain[1:],label="Pente 1/f")
        plt.legend()
        plt.savefig(path+"total_grain")
        plt.close()

        plt.title("Puissance spectrale du flux extérieur\n en fonction des fréquences")
        plt.xlabel("Fréquences (f) (Hz)"); plt.ylabel("S(f) (W^2/Hz)")
        plt.loglog(self.freqs_outflux[int((self.gen+1)*(4/10)):],self.ps_outflux[int((self.gen+1)*(4/10)):],label="Données")
        plt.loglog(self.freqs_outflux[int((self.gen+1)*(4/10)):],1/self.freqs_outflux[int((self.gen+1)*(4/10)):],label="Pente 1/f")
        plt.legend()
        plt.savefig(path+"outflux_PSD")
        plt.close()

        #SPATIAL CORRELATION (VALIDATION,~r^-4)
        plot_acf(self.corr_MEAN[:-10000],lags=np.arange(self.corr_MEAN[:-10000].size))
        plt.title("Auto-corrélation temporelle \n du nombre de grains par site")
        plt.xlabel("Période (t)"); plt.ylabel("Auto-corrélation C(t)")
        plt.savefig(path+"auto_correlation")
        plt.close()

if __name__=="__main__":
    N=50
    nb_sim=1
    nb_gen=50000
    Sim=Abelian_sandpile(N,nb_sim,nb_gen)
    freq_grain,ps_grain,avalanche_range,avalanche_dist=Sim.start_sim()
   

    from scipy.optimize import curve_fit
    inverse_avalanche_range=1/avalanche_range
    avalanche_slope,avalanche_cov=curve_fit(lambda x,a,b: a*x**b,inverse_avalanche_range,avalanche_dist,p0=(avalanche_dist[0],1))

    inverse_freqs_grain=1/freq_grain
    total_grain_slope,total_grain_cov=curve_fit(lambda x,a,b:a*x**b,inverse_freqs_grain,ps_grain,p0=(ps_grain[0],1))
    print(avalanche_slope,total_grain_slope)


    plt.title("Distribution de la taille des avalanches\n (As)")
    plt.loglog(avalanche_range,avalanche_dist,label="Données")
    plt.plot(avalanche_range,avalanche_slope[0]/avalanche_range**avalanche_slope[1],label=f'a={avalanche_slope[1]}±{np.sqrt(np.diag(avalanche_cov))[1]}')
    plt.legend()
    plt.savefig('As_sandpile/avalanche_size_fit')
    plt.close()

    plt.title("Puissance spectrale du nombre de grain de sable\n en fonction des fréquences\n (AS)")
    plt.xlabel("Fréquences (f) (Hz)"); plt.ylabel("S(f) (W^2/Hz)")
    plt.loglog(freq_grain,ps_grain,label="Données")
    plt.plot(freq_grain,total_grain_slope[0]/freq_grain**total_grain_slope[1],label=f'a={total_grain_slope[1]}±{np.sqrt(np.diag(total_grain_cov))[1]}')
    plt.legend()
    plt.savefig("As_sandpile/total_grain_fit")
    plt.close()
