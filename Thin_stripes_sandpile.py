import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf

class Thin_stripes_sandpile():

    def __init__(self,Nx,Ny,nb_sim,nb_gen,directory) -> None:
        print("INIT...",end="")
        #INIT LATTICE
        self.rng=np.random.default_rng()#flat distribution for adding slope

        self.size_x=Nx
        self.size_y=Ny

        self.lattice=np.zeros((self.size_y+2,self.size_x+2)) #init_lattice
        self.crit_height=2 #cricical point

        #ITERATIONS AND BURNING
        self.sim=nb_sim
        self.gen=nb_gen
        self.burn=self.gen//3

        #INIT QUANTITIES
        self.total_grain=np.array([])
        self.outflux=np.array([])

        self.corr3=np.array([])
        self.corr5=np.array([])
        
        self.dir=directory
        print("Done.")

    def start_sim(self):
        """
        start sim with n gen, calculate required observables
        """        
        #INIT MULTI-SIM OBSERVABLES
        self.avalanche_size=np.zeros((self.lattice.size*10))
        self.total_grain_MEAN=np.zeros((self.gen+1))
        self.outflux_MEAN=np.zeros((self.gen+1))

        #Loop of m simulations
        for m in range(self.sim):
            print(f"sim n.{m}")
            self.__init__(self.size_x,self.size_y,self.sim,self.gen,self.dir)

            #MAIN LOOP
            for n in range(self.gen+1):
                self.count=n
                coord=[self.rng.integers(1,self.size_y,1),1]#random coord    #coord=[self.size//2,self.size//2]

                #add slope unit and update quantities
                self.add_slope(coord)
                self.update_gen()
                print(self.count,end="\r")

            self.update_sim()

        self.total_grain_MEAN=self.total_grain_MEAN/(self.sim*np.max(self.total_grain_MEAN))
        self.outflux_MEAN=self.outflux_MEAN/(self.sim*np.max(self.outflux_MEAN))
        
        self.ps_grain,self.freqs_grain=self.PSD(self.total_grain_MEAN)

        self.plot()
        return self.freqs_grain[int((self.gen+1)*(4/10)):],self.ps_grain[int((self.gen+1)*(4/10)):],np.arange(10,self.avalanche_size.size-100),self.avalanche_size[10:-100]
    def update_gen(self):

        #check for critical heights
        if np.max(self.lattice)>self.crit_height:
            #topples critical point in while loop
            self.toppling()
            #update quantities
            self.outflux=np.append(self.outflux,self.outflux_micro)
            self.avalanche_size[np.clip(self.avalanche_size_micro,0,self.lattice.size*10-1)]+=1


        else: #if no critical points, update quantities
            self.outflux=np.append(self.outflux,0)
            self.avalanche_size[0]+=1

        self.total_grain=np.append(self.total_grain, np.sum(self.lattice))

        if self.count>=self.burn:
            self.corr3=np.append(self.corr3,np.sum(self.lattice[1:-1,4]))
            self.corr5=np.append(self.corr5,np.sum(self.lattice[1:-1,6]))
    
    def update_sim(self):

        self.total_grain_MEAN+=self.total_grain
        self.outflux_MEAN+=self.outflux
    
    def toppling(self):
        #observables temporaires
        self.outflux_micro=0
        self.avalanche_size_micro=0

        while np.max(self.lattice)>=self.crit_height: #this loop is 1 unit of micro-time
            crit_slope= self.lattice>=self.crit_height
                
            self.lattice-=self.toppling_matrix(crit_slope) #with crit_slope the coords with height>=crit_height


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

            #add all critical site to avalanche size
            self.avalanche_size_micro+=np.where(crit_slope==True)[0].size



    def toppling_matrix(self,coords):
        """
        adjacency matrix of the lattice at coord(s) (i1,j1), (i2,j2)...(im,jm)
        m=#{site with height>=crit_height}
        """
        adj_matrix=np.zeros((self.size_y+2,self.size_x+2))

        # adj_matrix_nn=2
        adj_matrix[coords] += self.crit_height
        
        # adj_matrix_n'n=-1 for 3 n' nearest neighbors
        adj_matrix[:,1:][coords[:,:-1]] -= self.crit_height/2
        
        #choose between up or down
        rdm=self.rng.choice([-1,1],coords.shape)
        rdm[np.where(coords==False)]=0
        coord_up=coords.copy()
        coord_up[np.where(rdm==-1)]=False
        coord_down=coords.copy()
        coord_down[np.where(rdm==1)]=False
        
        adj_matrix[1:,:][coord_up[:-1,:]] -= self.crit_height/2
        adj_matrix[:-1,:][coord_down[1:,:]] -= self.crit_height/2
        
    
        return adj_matrix
    def add_slope(self,coord):
        #add slope units
        self.lattice[coord[0],coord[1]]+=1
        

    ###COMPUTE QUANTITIES###        
    def PSD(self,observable):
        #code inspired from
        #https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python
        ps= np.abs(np.fft.fft(observable[self.burn:]))[1:]**2
        freqs= np.fft.fftfreq(observable[self.burn:].size)[1:]
        idx=np.argsort(freqs)

        return ps[idx],freqs[idx]
    ###PLOT###    
    def plot(self):
        
        path=os.path.join(os.path.dirname(os.path.abspath(__file__)),self.dir)
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

        plot_acf(self.corr3,lags=20)
        plt.title("Auto-corrélation temporelle \n du nombre de grains sur la colonne 3")
        plt.xlabel("Période (t)"); plt.ylabel("Auto-corrélation C(t)")
        plt.savefig(path+"auto_correlation_local_grain_3")
        plt.close()

        plot_acf(self.corr5,lags=20)
        plt.title("Auto-corrélation temporelle \n du nombre de grains sur la colonne 5")
        plt.xlabel("Période (t)"); plt.ylabel("Auto-corrélation C(t)")
        plt.savefig(path+"auto_correlation_local_grain_5")
        plt.close()


if __name__=="__main__":
    Nx=50
    Ny=4
    nb_sim,nb_gen=2,100000
    Sim=Thin_stripes_sandpile(Nx,Ny,nb_sim,nb_gen,"Ts_sandpile")
    freq_grain,ps_grain,avalanche_range,avalanche_dist=Sim.start_sim()
   

    from scipy.optimize import curve_fit

    inverse_freqs_grain=1/freq_grain
    total_grain_slope,total_grain_cov=curve_fit(lambda x,a,b:a*x**b,inverse_freqs_grain,ps_grain,p0=(ps_grain[0],1))
    print(total_grain_slope,np.sqrt(np.diag(total_grain_cov)))

    plt.title("Puissance spectrale du nombre de grain de sable\n en fonction des fréquences\n (NS)")
    plt.xlabel("Fréquences (f) (Hz)"); plt.ylabel("S(f) (W^2/Hz)")
    plt.loglog(freq_grain,ps_grain,label="Données")
    plt.plot(freq_grain,total_grain_slope[0]/freq_grain**total_grain_slope[1],label=f'a={total_grain_slope[1]}±{np.sqrt(np.diag(total_grain_cov))[1]}')
    plt.legend()
    plt.savefig("Ts_sandpile/total_grain_fit")
    plt.close() 

        #inverse avalanche to fit curve better
    inverse_avalanche_range=1/avalanche_range
    #find exponent
    avalanche_slope,avalanche_cov=curve_fit(lambda x,a,b: a*x**b,inverse_avalanche_range,avalanche_dist,p0=(avalanche_dist[0],1))

    plt.title("Distribution de la taille des avalanches\n (As)")
    plt.xlabel("Taille des avalanches (s)"); plt.ylabel("P(s)")
    plt.loglog(avalanche_range,avalanche_dist,label="Données")
    plt.plot(avalanche_range,avalanche_slope[0]/avalanche_range**avalanche_slope[1],label=f'a={np.around(avalanche_slope[1],3)}±{np.around(np.sqrt(np.diag(avalanche_cov))[1],3)}')
    plt.legend()
    plt.savefig('Ts_sandpile_big/avalanche_size_fit')
    plt.close()