import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Abelian_sandpile import Abelian_sandpile
from Dissipative_sandpile import Continuous_dissipative_sandpile
from Thin_stripes_sandpile import Thin_stripes_sandpile

#INIT SIM 
N=50


### ABELIAN SANDPILE
print("### ABELIAN SANDPILE")
nb_sim=10
nb_gen=100000
SimA=Abelian_sandpile(N,nb_sim,nb_gen)

#f, PSD(f) , s, P(s)
freq_grain,ps_grain,avalanche_range,avalanche_dist=SimA.start_sim()

#inverse avalanche to fit curve better
inverse_avalanche_range=1/avalanche_range
#find exponent
avalanche_slope,avalanche_cov=curve_fit(lambda x,a,b: a*x**b,inverse_avalanche_range,avalanche_dist,p0=(avalanche_dist[0],1))

#inverse frequence to fit curve better
inverse_freqs_grain=1/freq_grain
#find exponent
total_grain_slope,total_grain_cov=curve_fit(lambda x,a,b:a*x**b,inverse_freqs_grain,ps_grain,p0=(ps_grain[0],1))

#PLOT
plt.title("Distribution de la taille des avalanches\n (As)")
plt.xlabel("Taille des avalanches (s)"); plt.ylabel("P(s)")
plt.loglog(avalanche_range,avalanche_dist,label="Données")
plt.plot(avalanche_range,avalanche_slope[0]/avalanche_range**avalanche_slope[1],label=f'a={np.around(avalanche_slope[1],3)}±{np.around(np.sqrt(np.diag(avalanche_cov))[1],3)}')
plt.legend()
plt.savefig('As_sandpile/avalanche_size_fit')
plt.close()

plt.title("Puissance spectrale du nombre de grain de sable\n en fonction des fréquences\n (AS)")
plt.xlabel("Fréquences (f) (Hz)"); plt.ylabel("S(f) (W^2/Hz)")
plt.loglog(freq_grain,ps_grain,label="Données")
plt.plot(freq_grain,total_grain_slope[0]/freq_grain**total_grain_slope[1],label=f'a={np.around(total_grain_slope[1],3)}±{np.around(np.sqrt(np.diag(total_grain_cov))[1],3)}')
plt.legend()
plt.savefig("As_sandpile/total_grain_fit")
plt.close()

### DISSIPATIVE SANDPILE
print("### DISSIPATIVE SANDPILE")
nb_sim=10
nb_gen=50000

SimD=Continuous_dissipative_sandpile(N,nb_sim,nb_gen)

#f, PSD(f) , s, P(s)
freq_grain,ps_grain,avalanche_range,avalanche_dist=SimD.start_sim()

#inverse to fit exponent better
inverse_freqs_grain=1/freq_grain
#find exponent
total_grain_slope,total_grain_cov=curve_fit(lambda x,a,b:a*x**b,inverse_freqs_grain,ps_grain,p0=(ps_grain[0],1))

plt.title("Puissance spectrale du nombre de grain de sable\n en fonction des fréquences\n (DS)")
plt.xlabel("Fréquences (f) (Hz)"); plt.ylabel("S(f) (W^2/Hz)")
plt.loglog(freq_grain,ps_grain,label="Données")
plt.plot(freq_grain,total_grain_slope[0]/freq_grain**total_grain_slope[1],label=f'a={np.around(total_grain_slope[1],3)}±{np.around(np.sqrt(np.diag(total_grain_cov))[1],3)}')
plt.legend()
plt.savefig("Ds_sandpile/total_grain_fit")
plt.close()
        
### THIN STRIPES SANDPILE
print("### THIN STRIPES SANDPILE")
nb_sim=10
nb_gen=100000
Nx=8
Ny=2
directory="Ts_sandpile"
SimT=Thin_stripes_sandpile(Nx,Ny,nb_sim,nb_gen,directory)

#f, PSD(f) for total grain Z(t)
freq_grain,ps_grain.avalanche_range,avalanche_dist=SimT.start_sim()

inverse_freqs_grain=1/freq_grain
total_grain_slope,total_grain_cov=curve_fit(lambda x,a,b:a*x**b,inverse_freqs_grain,ps_grain,p0=(ps_grain[0],1))

plt.title("Puissance spectrale du nombre de grain de sable\n en fonction des fréquences\n (TS)")
plt.xlabel("Fréquences (f) (Hz)"); plt.ylabel("S(f) (W^2/Hz)")
plt.loglog(freq_grain,ps_grain,label="Données")
plt.plot(freq_grain,total_grain_slope[0]/freq_grain**total_grain_slope[1],label=f'a={np.around(total_grain_slope[1],3)}±{np.around(np.sqrt(np.diag(total_grain_cov))[1],3)}')
plt.legend()
plt.savefig("Ts_sandpile/total_grain_fit")
plt.close()

### BIG STRIPES SANDPILE
print("### BIG STRIPES SANDPILE")
nb_sim=10
nb_gen=100000
Nx=50
Ny=4
directory="Ts_sandpile_big"
SimT=Thin_stripes_sandpile(Nx,Ny,nb_sim,nb_gen,directory)

#f, PSD(f) for total grain Z(t) and s P(s) for avalanche dist
freq_grain,ps_grain,avalanche_range,avalanche_dist=SimT.start_sim()

inverse_freqs_grain=1/freq_grain
total_grain_slope,total_grain_cov=curve_fit(lambda x,a,b:a*x**b,inverse_freqs_grain,ps_grain,p0=(ps_grain[0],1))

plt.title("Puissance spectrale du nombre de grain de sable\n en fonction des fréquences\n (TS)")
plt.xlabel("Fréquences (f) (Hz)"); plt.ylabel("S(f) (W^2/Hz)")
plt.loglog(freq_grain,ps_grain,label="Données")
plt.plot(freq_grain,total_grain_slope[0]/freq_grain**total_grain_slope[1],label=f'a={np.around(total_grain_slope[1],3)}±{np.around(np.sqrt(np.diag(total_grain_cov))[1],3)}')
plt.legend()
plt.savefig("Ts_sandpile_big/total_grain_fit")
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
plt.savefig('As_sandpile/avalanche_size_fit')
plt.close()