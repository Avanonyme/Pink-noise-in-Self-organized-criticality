import numpy as np
import matplotlib.pyplot as plt
import os
#### ABELIAN_SANDPILE


class Abelian_sandpile:

    def __init__(self,N,nb_sim,nb_gen) -> None:
        print("INIT...",end="")
        #INIT LATTICE
        self.rng=np.random.default_rng()#flat distribution for adding slope

        self.size=N #L=NxN
        self.lattice=np.ones((self.size+1,self.size+1))*3 #init_lattice
        self.crit_height=4 #cricical point

        #ITERATIONS AND Burning
        self.sim=nb_sim
        self.gen=nb_gen
        self.burn=self.gen//4

        fig, ax = plt.subplots()
        bar=plt.colorbar(plt.imshow(self.lattice,vmin=0,vmax=4))
        print("Done.")

    def start_sim(self):
        """
        start sim with n gen, calculate required observables
        """        
        #INIT MULTI-SIM OBSERVABLES
        self.avalanche_size=np.zeros((self.lattice.size*10))
        self.total_grain_MEAN=np.zeros((self.gen+1))
        self.outflux_MEAN=np.zeros((self.gen+1))

        #MAIN LOOP
        for n in range(self.gen+1):
            self.count=n
            coord=[self.size//2,self.size//2]#random coord    #coord=[self.size//2,self.size//2]

            #add slope unit and update quantities
            self.add_slope(coord)
            self.update_gen()
            print(self.count,end="\r")
            #update on all sim
        return None
    def update_gen(self):

        #check for critical heights
        if np.max(self.lattice)>=self.crit_height:
            #topples critical point in while loop
            self.toppling()

    def toppling(self):
        while np.max(self.lattice)>=self.crit_height: #this loop is 1 unit of micro-time
             crit_slope= self.lattice>=self.crit_height
                
             self.lattice-=self.toppling_matrix(crit_slope) #with crit_slope the coords with height>=crit_height

            #open boundaries
             self.lattice[0,:] = 0
             self.lattice[self.size,:] = 0
             self.lattice[:,0] = 0
             self.lattice[:,self.size] = 0
             self.plot(self.lattice)

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

    def plot(self,lattice):
        plt.imshow(lattice,vmin=0,vmax=4)
        plt.pause(2)

class Continuous_dissipative_sandpile:


    def __init__(self,N,nb_sim,nb_gen) -> None:
        print("INIT...",end="")
        #INIT LATTICE
        self.rng=np.random.default_rng()#flat distribution for adding slope

        self.size=N #L=NxN
        self.lattice=np.ones((self.size+1,self.size+1))-0.1 #init_lattice
        self.lattice[0:,0]=0

        self.crit_E=1 #critical point
        self.dissipation=0.01 #dissipation
        self.coordination=4

        #ITERATIONS AND Burning
        self.sim=nb_sim
        self.gen=nb_gen
        self.burn=self.gen//4

        fig, ax = plt.subplots()
        bar=plt.colorbar(plt.imshow(self.lattice,vmin=0,vmax=4))

        print("Done.")
    
    ###MAIN###
    def start_sim(self):
        """
        start sim with n gen, calculate required observables
        """        
            #MAIN LOOP
        for n in range(self.gen+1):
            self.count=n
            coord=(self.rng.integers(0,self.size),1)

            self.add_energy(coord)
            self.update_gen()
            print(self.count,"\r",end="")


        return None
    
    ###UPDATES###
    def update_gen(self):

        #check for critical energy
        if np.max(self.lattice)>=self.crit_E:
            #topples critical point in while loop
            self.toppling()
        

    def neighbors_energy(self,coord):
        try:
            self.lattice[1:,:][coord[:-1,:]] += self.lattice[coord]/self.coordination*(1-self.dissipation) #up
            self.lattice[:-1,:][coord[1:,:]] += self.lattice[coord]/self.coordination*(1-self.dissipation) #down
            self.lattice[:,1:][coord[:,:-1]] += self.lattice[coord]/self.coordination*(1-self.dissipation) #left
            self.lattice[:,:-1][coord[:,1:]] += self.lattice[coord]/self.coordination*(1-self.dissipation) #right

        #Handle unresolved errors at down boundary
        except:
            self.lattice[-1,1:]=0
            self.outflux_micro+=1
            pass

    def add_energy(self,coord):
        eps=self.rng.random()*10
        self.lattice[coord]+=eps

    def toppling(self):
        self.outflux_micro=0
        self.avalanche_size_micro=0

        while np.max(self.lattice)>=self.crit_E:
            n_crit= self.lattice>=self.crit_E

            self.neighbors_energy(n_crit)
            self.lattice[n_crit]=0
            self.plot(self.lattice)
            #reset boundaries

            self.lattice[0,1:]=0
            self.lattice[-1,1:]=0
            self.lattice[0:,-1]=0

            #closed boundary on the left side
            self.lattice[0:,1]+=self.lattice[0:,0]
            self.lattice[0:,0]=0


    def plot(self,lattice):
        plt.imshow(lattice,vmin=0,vmax=4)
        plt.pause(2)

class Thin_stripes_sandpile():

    def __init__(self,Nx,Ny,nb_gen) -> None:
        print("INIT...",end="")
        #INIT LATTICE
        self.rng=np.random.default_rng()#flat distribution for adding slope

        self.size_x=Nx
        self.size_y=Ny

        self.lattice=np.ones((self.size_y+2,self.size_x+2)) #init_lattice
        #Resset boundaries
        self.lattice[0:,0]=0
        self.lattice[0,:]=0
        self.lattice[-1,:]=0

        self.crit_height=2 #cricical point

        #ITERATIONS AND BURNING
        self.gen=nb_gen
        self.burn=self.gen//4

        fig, ax = plt.subplots()
        bar=plt.colorbar(plt.imshow(self.lattice,vmin=0,vmax=2))

        print("Done.")

    def start_sim(self):
        """
        start sim with n gen, calculate required observables
        """       

        #MAIN LOOP
        for n in range(self.gen+1):
            self.count=n
            coord=[self.rng.integers(1,self.size_y,1),1]#random cord

            #add slope unit and update quantities
            self.add_slope(coord)
            self.update_gen()
            print(self.count,end="\r")


        return None
    def update_gen(self):

        #check for critical heights
        if np.max(self.lattice)>self.crit_height:
            #topples critical point in while loop
            self.toppling()
    
    def toppling(self):
        #observables temporaires
        self.outflux_micro=0
        self.avalanche_size_micro=0

        while np.max(self.lattice)>=self.crit_height: #this loop is 1 unit of micro-time
            crit_slope= self.lattice>=self.crit_height
                
            self.lattice-=self.toppling_matrix(crit_slope) #with crit_slope the coords with height>=crit_height
 
            self.plot(self.lattice)

            #closed boundary on the left-x side and both y sides
            self.lattice[0:,1]+=self.lattice[0:,0]
            self.lattice[0:,0]=0
            
            self.lattice[1,:]+=self.lattice[0,:]
            self.lattice[0,:]=0

            self.lattice[-2,:]+=self.lattice[-1,:]
            self.lattice[-1,:]=0

            #reset boundaries
            self.lattice[0:,-1]=0

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

    def plot(self,lattice):
        self.count+=1
        plt.imshow(lattice,vmin=0,vmax=2)
        plt.pause(2)
        plt.savefig(f"gravity_check{self.count}")

if __name__=="__main__":

    try:
        Nx=8
        Ny=2
        N=50
        nb_sim=1
        nb_gen=1
        choice=int(input("Input the number:\nAbelian(1),Dissipative(2) or Thin Stripes(3)"))

        if choice==1:
            Sim=Abelian_sandpile(N,nb_sim,nb_gen)
            Sim.start_sim()
        if choice==2:
            Sim=Continuous_dissipative_sandpile(N,nb_sim,nb_gen)
            Sim.start_sim()
        if choice==3:
            Sim=Thin_stripes_sandpile(Nx,Ny,nb_gen)
            Sim.start_sim()

        else:
            print("NOT IN OPTION\n")

    except KeyboardInterrupt:print("")