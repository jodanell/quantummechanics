#Import the important modules
import numpy as np
import matplotlib.pyplot as plt

#Create arrays for atomic numbers (Z) and corresponding Shannon entropy values (S) from the paper
Z_paper = np.arange(1, 55)
S_paper = np.array([
    6.56659, 6.61193, 7.69826, 7.81405, 8.11135, 8.26260, 8.35103, 8.41791, 8.46215, 8.49221,
    8.81319, 8.91038, 9.06497, 9.15294, 9.20767, 9.24871, 9.27418, 9.28924, 9.47419, 9.54334,
    9.60143, 9.64548, 9.68229, 9.70724, 9.73945, 9.76478, 9.78613, 9.71120, 9.81401, 9.83493,
    9.89832, 9.93896, 9.96808, 9.99276, 10.0102, 10.0224, 10.1206, 10.1637, 10.2048, 10.2352,
    10.2572, 10.2735, 10.2984, 10.3061, 10.3184, 10.2939, 10.3374, 10.3574, 10.4008, 10.43,
    10.4517, 10.4706, 10.4849, 10.496
])

#Fit the paper's data to a logarithmic relationship
#Take natural log of atomic numbers
logZ_paper = np.log(Z_paper)
#Perform linear regression on ln(Z) vs S
p_log = np.polyfit(logZ_paper, S_paper, 1)
#Generate fitted values using the regression coefficients
S_logfit = np.polyval(p_log, logZ_paper)

#RHF data - more carefully interpreted
#Dictionary containing STO coefficients, zetas, and orbital counts
rhf_sto = {
    2: ([0.7683, 0.3130], [1.5750, 3.5730], [2]),
    3: ([0.5090, 0.3365, 0.1435], [2.4500, 5.5500, 0.6530], [2, 1]),
    4: ([0.7356, 0.2743, 0.0687], [3.1450, 7.2750, 1.0500], [2, 2]),
    5: ([0.7061, 0.2992, 0.0973, 0.0393], [3.7400, 8.7650, 1.1700, 0.3820], [2, 2, 1]),
    6: ([0.6857, 0.3159, 0.1137, 0.0465, 0.0197], [4.3350, 10.2550, 1.2900, 0.4240, 0.1500], [2, 2, 2]),
    7: ([0.6714, 0.3274, 0.1229, 0.0513, 0.0214, 0.0092], [4.9300, 11.7450, 1.4100, 0.4660, 0.1650, 0.0600], [2, 2, 3]),
    8: ([0.6602, 0.3357, 0.1297, 0.0547, 0.0229, 0.0100, 0.0043], [5.5250, 13.2350, 1.5300, 0.5080, 0.1800, 0.0660, 0.0250], [2, 2, 4]),
    9: ([0.6515, 0.3423, 0.1349, 0.0576, 0.0242, 0.0106, 0.0046, 0.0020], [6.1200, 14.7250, 1.6500, 0.5500, 0.1950, 0.0720, 0.0280, 0.0100], [2, 2, 5]),
    10: ([0.6549, 0.3477, 0.1391, 0.0598, 0.0252, 0.0110, 0.0049, 0.0021, 0.0009], [6.7150, 16.2150, 1.7700, 0.5920, 0.2100, 0.0780, 0.0310, 0.0110, 0.0040], [2, 2, 6])
}

ionization_potentials = {
    2: 24.587,   # He
    3: 5.392,    # Li
    4: 9.323,    # Be
    5: 8.298,    # B
    6: 11.260,   # C
    7: 14.534,   # N
    8: 13.618,   # O
    9: 17.423,   # F
    10: 21.565   # Ne
}

#Print header for the results table
print(f"{'Z':<3} {'Sr':<8} {'Sk':<8} {'S':<8} {'Or':<8} {'Ok':<8} {'Fr':<8} {'Fk':<8} {'O_total':<8} {'F_total':<8}")
print("_" * 80)

#Create dictionary to store calculation results
results = {
    'Z': [], 'Sr': [], 'Sk': [], 'S': [],
    'Or': [], 'Ok': [], 'Fr': [], 'Fk': [], 'O_total': [], 'F_total': []
}

#Iterate through atomic numbers from 2 to 10
for Z in range(2, 11):
    #Define orbital structure for each element
    if Z == 2:  #He: 1s²
        orbital_structure = [('1s', 2)]
    elif Z == 3:  #Li: 1s² 2s¹
        orbital_structure = [('1s', 2), ('2s', 1)]
    elif Z == 4:  #Be: 1s² 2s²
        orbital_structure = [('1s', 2), ('2s', 2)]
    elif Z == 5:  #B: 1s² 2s² 2p¹
        orbital_structure = [('1s', 2), ('2s', 2), ('2p', 1)]
    elif Z == 6:  #C: 1s² 2s² 2p²
        orbital_structure = [('1s', 2), ('2s', 2), ('2p', 2)]
    elif Z == 7:  #N: 1s² 2s² 2p³
        orbital_structure = [('1s', 2), ('2s', 2), ('2p', 3)]
    elif Z == 8:  #O: 1s² 2s² 2p⁴
        orbital_structure = [('1s', 2), ('2s', 2), ('2p', 4)]
    elif Z == 9:  #F: 1s² 2s² 2p⁵
        orbital_structure = [('1s', 2), ('2s', 2), ('2p', 5)]
    elif Z == 10:  #Ne: 1s² 2s² 2p⁶
        orbital_structure = [('1s', 2), ('2s', 2), ('2p', 6)]
    else:
        continue
    
    #Get STO coefficients and zetas from RHF data
    coeffs, zetas, _ = rhf_sto[Z]
    #Create dictionary to store effective nuclear charge for each orbital type
    zeff_dict = {}
    #Assign zeta values to orbital types
    for orbital_type, _ in orbital_structure:
        if orbital_type not in zeff_dict:
            zeff_dict[orbital_type] = zetas[0] if zetas else 1.0
    
    #Calculate normalization constants
    try:
        #Position space normalization - integrate 4πr^2 * ρ(r) from 0 to inf
        norm_r_integrand = []
        #Sample radial points for numerical integration
        for r in np.linspace(1e-6, 20, 1000):
            result = 0
            #Calculate electron density at each point
            for orbital_type, occupation in orbital_structure:
                if orbital_type in zeff_dict:
                    zeff = zeff_dict[orbital_type]
                    #STO orbital functions (position space)
                    if orbital_type == '1s':
                        psi = np.sqrt(zeff**3 / np.pi) * np.exp(-zeff * r)
                    elif orbital_type == '2s':
                        psi = np.sqrt(zeff**5 / (3 * np.pi)) * r * np.exp(-zeff * r)
                    elif orbital_type == '2p':
                        psi = np.sqrt(zeff**5 / (3 * np.pi)) * r * np.exp(-zeff * r)
                    else:
                        psi = 0
                    #Sum contributions from all orbitals
                    result += occupation * psi**2
            #Store integrand value
            norm_r_integrand.append(result * r**2)
        
        #Perform numerical integration
        norm_r = 4 * np.pi * np.trapz(norm_r_integrand, np.linspace(1e-6, 20, 1000))
        
        #Momentum space normalization - integrate 4πk^2 * ρ(k) from 0 to inf
        norm_k_integrand = []
        #Sample momentum points for numerical integration
        for k in np.linspace(1e-6, 20, 1000):
            result = 0
            #Calculate momentum density at each point
            for orbital_type, occupation in orbital_structure:
                if orbital_type in zeff_dict:
                    zeff = zeff_dict[orbital_type]
                    #STO orbital functions (momentum space)
                    if orbital_type == '1s':
                        psi = np.sqrt(8 * np.pi) * zeff**2.5 / (k**2 + zeff**2)**2
                    elif orbital_type == '2s':
                        psi = np.sqrt(8 * np.pi) * 2 * zeff**3.5 / (k**2 + zeff**2)**3
                    elif orbital_type == '2p':
                        psi = np.sqrt(8 * np.pi) * zeff**3.5 * k**2 / (k**2 + zeff**2)**3
                    else:
                        psi = 0
                    #Sum contributions from all orbitals
                    result += occupation * psi**2
            #Store integrand value
            norm_k_integrand.append(result * k**2)
        
        #Perform numerical integration
        norm_k = 4 * np.pi * np.trapz(norm_k_integrand, np.linspace(1e-6, 20, 1000))
        
        #Check for invalid normalization
        if norm_r <= 0 or norm_k <= 0:
            continue
        
        #Calculate Shannon entropy in position space
        shannon_r_integrand = []
        for r in np.linspace(1e-6, 20, 1000):
            rho = 0
            #Calculate electron density
            for orbital_type, occupation in orbital_structure:
                if orbital_type in zeff_dict:
                    zeff = zeff_dict[orbital_type]
                    if orbital_type == '1s':
                        psi = np.sqrt(zeff**3 / np.pi) * np.exp(-zeff * r)
                    elif orbital_type == '2s':
                        psi = np.sqrt(zeff**5 / (3 * np.pi)) * r * np.exp(-zeff * r)
                    elif orbital_type == '2p':
                        psi = np.sqrt(zeff**5 / (3 * np.pi)) * r * np.exp(-zeff * r)
                    else:
                        psi = 0
                    rho += occupation * psi**2
            
            #Calculate probability density
            p = rho / norm_r
            if p <= 1e-15:
                shannon_r_integrand.append(0)
            else:
                shannon_r_integrand.append(p * np.log(p) * r**2)
        
        #Calculate Shannon entropy by integration
        Sr = -4 * np.pi * np.trapz(shannon_r_integrand, np.linspace(1e-6, 20, 1000))
        
        #Calculate Shannon entropy in momentum space
        shannon_k_integrand = []
        for k in np.linspace(1e-6, 20, 1000):
            rho = 0
            #Calculate momentum density
            for orbital_type, occupation in orbital_structure:
                if orbital_type in zeff_dict:
                    zeff = zeff_dict[orbital_type]
                    if orbital_type == '1s':
                        psi = np.sqrt(8 * np.pi) * zeff**2.5 / (k**2 + zeff**2)**2
                    elif orbital_type == '2s':
                        psi = np.sqrt(8 * np.pi) * 2 * zeff**3.5 / (k**2 + zeff**2)**3
                    elif orbital_type == '2p':
                        psi = np.sqrt(8 * np.pi) * zeff**3.5 * k**2 / (k**2 + zeff**2)**3
                    else:
                        psi = 0
                    rho += occupation * psi**2
            
            #Calculate probability density
            p = rho / norm_k
            if p <= 1e-15:
                shannon_k_integrand.append(0)
            else:
                shannon_k_integrand.append(p * np.log(p) * k**2)
        
        #Calculate Shannon entropy by integration
        Sk = -4 * np.pi * np.trapz(shannon_k_integrand, np.linspace(1e-6, 20, 1000))
        
        #Calculate Onicescu energy in position space
        onicescu_r_integrand = []
        for r in np.linspace(1e-6, 20, 1000):
            rho = 0
            #Calculate electron density
            for orbital_type, occupation in orbital_structure:
                if orbital_type in zeff_dict:
                    zeff = zeff_dict[orbital_type]
                    if orbital_type == '1s':
                        psi = np.sqrt(zeff**3 / np.pi) * np.exp(-zeff * r)
                    elif orbital_type == '2s':
                        psi = np.sqrt(zeff**5 / (3 * np.pi)) * r * np.exp(-zeff * r)
                    elif orbital_type == '2p':
                        psi = np.sqrt(zeff**5 / (3 * np.pi)) * r * np.exp(-zeff * r)
                    else:
                        psi = 0
                    rho += occupation * psi**2
            
            #Calculate probability density
            p = rho / norm_r
            onicescu_r_integrand.append(p**2 * r**2)
        
        #Calculate Onicescu energy by integration
        Or = 4 * np.pi * np.trapz(onicescu_r_integrand, np.linspace(1e-6, 20, 1000))
        
        #Calculate Onicescu energy in momentum space
        onicescu_k_integrand = []
        for k in np.linspace(1e-6, 20, 1000):
            rho = 0
            #Calculate momentum density
            for orbital_type, occupation in orbital_structure:
                if orbital_type in zeff_dict:
                    zeff = zeff_dict[orbital_type]
                    if orbital_type == '1s':
                        psi = np.sqrt(8 * np.pi) * zeff**2.5 / (k**2 + zeff**2)**2
                    elif orbital_type == '2s':
                        psi = np.sqrt(8 * np.pi) * 2 * zeff**3.5 / (k**2 + zeff**2)**3
                    elif orbital_type == '2p':
                        psi = np.sqrt(8 * np.pi) * zeff**3.5 * k**2 / (k**2 + zeff**2)**3
                    else:
                        psi = 0
                    rho += occupation * psi**2
            
            #Calculate probability density
            p = rho / norm_k
            onicescu_k_integrand.append(p**2 * k**2)
        
        #Calculate Onicescu energy by integration
        Ok = 4 * np.pi * np.trapz(onicescu_k_integrand, np.linspace(1e-6, 20, 1000))
        
        #Calculate Fisher information in position space
        fisher_r_integrand = []
        r_vals = np.linspace(1e-6, 20, 1000)
        for r in r_vals:
            rho = 0
            grad_rho = 0
            #Calculate electron density and its gradient
            for orbital_type, occupation in orbital_structure:
                if orbital_type in zeff_dict:
                    zeff = zeff_dict[orbital_type]
                    if orbital_type == '1s':
                        psi = np.sqrt(zeff**3 / np.pi) * np.exp(-zeff * r)
                        dpsi_dr = np.sqrt(zeff**3 / np.pi) * (-zeff) * np.exp(-zeff * r)
                    elif orbital_type == '2s':
                        psi = np.sqrt(zeff**5 / (3 * np.pi)) * r * np.exp(-zeff * r)
                        dpsi_dr = np.sqrt(zeff**5 / (3 * np.pi)) * (1 - zeff * r) * np.exp(-zeff * r)
                    elif orbital_type == '2p':
                        psi = np.sqrt(zeff**5 / (3 * np.pi)) * r * np.exp(-zeff * r)
                        dpsi_dr = np.sqrt(zeff**5 / (3 * np.pi)) * (1 - zeff * r) * np.exp(-zeff * r)
                    else:
                        psi = 0
                        dpsi_dr = 0
                    rho += occupation * psi**2
                    grad_rho += occupation * 2 * psi * dpsi_dr
            
            #Calculate probability density and its gradient
            p = rho / norm_r
            if p <= 1e-15:
                fisher_r_integrand.append(0)
            else:
                grad_p = grad_rho / norm_r
                fisher_r_integrand.append((grad_p**2 / p) * r**2)
        
        #Calculate Fisher information by integration
        Fr = 4 * np.pi * np.trapz(fisher_r_integrand, r_vals)
        
        #Calculate Fisher information in momentum space
        fisher_k_integrand = []
        k_vals = np.linspace(1e-6, 20, 1000)
        for k in k_vals:
            rho = 0
            grad_rho = 0
            #Calculate momentum density and its gradient
            for orbital_type, occupation in orbital_structure:
                if orbital_type in zeff_dict:
                    zeff = zeff_dict[orbital_type]
                    if orbital_type == '1s':
                        psi = np.sqrt(8 * np.pi) * zeff**2.5 / (k**2 + zeff**2)**2
                        dpsi_dk = np.sqrt(8 * np.pi) * zeff**2.5 * (-4 * k) / (k**2 + zeff**2)**3
                    elif orbital_type == '2s':
                        psi = np.sqrt(8 * np.pi) * 2 * zeff**3.5 / (k**2 + zeff**2)**3
                        dpsi_dk = np.sqrt(8 * np.pi) * 2 * zeff**3.5 * (-6 * k) / (k**2 + zeff**2)**4
                    elif orbital_type == '2p':
                        psi = np.sqrt(8 * np.pi) * zeff**3.5 * k**2 / (k**2 + zeff**2)**3
                        dpsi_dk = np.sqrt(8 * np.pi) * zeff**3.5 * (2 * k * (k**2 + zeff**2) - k**2 * 2 * k) / (k**2 + zeff**2)**4
                    else:
                        psi = 0
                        dpsi_dk = 0
                    rho += occupation * psi**2
                    grad_rho += occupation * 2 * psi * dpsi_dk
            
            #Calculate probability density and its gradient
            p = rho / norm_k
            if p <= 1e-15:
                fisher_k_integrand.append(0)
            else:
                grad_p = grad_rho / norm_k
                fisher_k_integrand.append((grad_p**2 / p) * k**2)
        
        #Calculate Fisher information by integration
        Fk = 4 * np.pi * np.trapz(fisher_k_integrand, k_vals)
        
        #Calculate total measures
        S = Sr + Sk
        
        #Store results in dictionary
        results['Z'].append(Z)
        results['Sr'].append(Sr)
        results['Sk'].append(Sk)
        results['S'].append(S)
        results['Or'].append(Or)
        results['Ok'].append(Ok)
        results['Fr'].append(Fr)
        results['Fk'].append(Fk)
        results['O_total'].append(Or + Ok)
        results['F_total'].append(Fr + Fk)
        
        #Print results for current element
        print(f"{Z:<3} {Sr:<8.4f} {Sk:<8.4f} {S:<8.4f} {Or:<8.4f} {Ok:<8.4f} {Fr:<8.4f} {Fk:<8.4f} {Or + Ok:<8.4f} {Fr + Fk:<8.4f}")
        
    except Exception as e:
        print(f"Integration failed for Z={Z}: {e}")
        continue

#Fit calculated data to logarithmic relationship
if len(results['Z']) > 0:
    #Extract calculated values
    Z_calc = np.array(results['Z'])
    S_calc = np.array(results['S'])
    
    #Fit data to logarithmic relationship
    logZ_calc = np.log(Z_calc)
    p_calc = np.polyfit(logZ_calc, S_calc, 1)
    S_calc_fit = np.polyval(p_calc, logZ_calc)
    
    #Calculate R^2 for this fit
    ss_res = np.sum((S_calc - S_calc_fit) ** 2)
    ss_tot = np.sum((S_calc - np.mean(S_calc)) ** 2)
    r_squared_calc = 1 - (ss_res / ss_tot)
    
    #Calculate R² for paper fit
    ss_res_paper = np.sum((S_paper - S_logfit) ** 2)
    ss_tot_paper = np.sum((S_paper - np.mean(S_paper)) ** 2)
    r_squared_paper = 1 - (ss_res_paper / ss_tot_paper)

#Create comprehensive comparison figure
fig = plt.figure(figsize=(16, 12))

#Main information measures plot
ax1 = plt.subplot(2, 3, 1)
ax1.plot(results['Z'], results['Sr'], 'o-', label='Sr (position)', linewidth=2, markersize=6)
ax1.plot(results['Z'], results['Sk'], 's-', label='Sk (momentum)', linewidth=2, markersize=6)
ax1.plot(results['Z'], results['S'], '^-', label='S = Sr + Sk', linewidth=2, color='red', markersize=6)
ax1.set_xlabel('Atomic Number Z')
ax1.set_ylabel('Shannon Entropy')
ax1.set_title('(a) Current Calculated Shannon Entropy')
ax1.legend()
ax1.grid(True, alpha=0.3)

#Paper's data and fit
ax2 = plt.subplot(2, 3, 2)
ax2.plot(Z_paper, S_paper, 'ko', markersize=4, label='Paper Data')
ax2.plot(Z_paper, S_logfit, 'b--', linewidth=2, 
         label=f'Log Fit: S = {p_log[1]:.3f} + {p_log[0]:.3f} ln(Z)')
ax2.set_xlabel('Atomic Number Z')
ax2.set_ylabel('Total Entropy S')
ax2.set_title('(b) Paper Data and Logarithmic Fit')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(np.arange(0, 55, 10))
ax2.set_xlim(0, 55)
ax2.text(5, 10.5, f"Paper's relationship:\nS = 6.257 + 1.069 ln(Z)\nR² = {r_squared_paper:.4f}", 
         fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

#Current data fitted to logarithmic relationship
ax3 = plt.subplot(2, 3, 3)
if len(results['Z']) > 0:
    ax3.plot(Z_calc, S_calc, 'ro', markersize=6, label='Current Calculated Data')
    ax3.plot(Z_calc, S_calc_fit, 'g--', linewidth=2, 
             label=f'Current Log Fit: S = {p_calc[1]:.3f} + {p_calc[0]:.3f} ln(Z)')
    ax3.set_xlabel('Atomic Number Z')
    ax3.set_ylabel('Total Entropy S')
    ax3.set_title('(c) Current Data and Logarithmic Fit')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.text(2.5, max(S_calc) * 0.95, f"Current relationship:\nS = {p_calc[1]:.3f} + {p_calc[0]:.3f} ln(Z)\nR² = {r_squared_calc:.4f}", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

#Onicescu Energy
ax4 = plt.subplot(2, 3, 4)
ax4.plot(results['Z'], results['Or'], 'o-', label='Or (position)', linewidth=2, markersize=6)
ax4.plot(results['Z'], results['Ok'], 's-', label='Ok (momentum)', linewidth=2, markersize=6)
ax4.plot(results['Z'], results['O_total'], '^-', label='O = Or + Ok', linewidth=2, color='red', markersize=6)
ax4.set_xlabel('Atomic Number Z')
ax4.set_ylabel('Onicescu Energy')
ax4.set_title('(d) Onicescu Energy')
ax4.legend()
ax4.grid(True, alpha=0.3)

#Fisher Information
ax5 = plt.subplot(2, 3, 5)
ax5.plot(results['Z'], results['Fr'], 'o-', label='Fr (position)', linewidth=2, markersize=6)
ax5.plot(results['Z'], results['Fk'], 's-', label='Fk (momentum)', linewidth=2, markersize=6)
ax5.plot(results['Z'], results['F_total'], '^-', label='F = Fr + Fk', linewidth=2, color='red', markersize=6)
ax5.set_xlabel('Atomic Number Z')
ax5.set_ylabel('Fisher Information')
ax5.set_title('(e) Fisher Information')
ax5.legend()
ax5.grid(True, alpha=0.3)

#Comparison of all total measures
ax6 = plt.subplot(2, 3, 6)
ax6.plot(results['Z'], results['S'], 'o-', label='Shannon Entropy', linewidth=2, markersize=6)
ax6.plot(results['Z'], results['O_total'], 's-', label='Onicescu Energy', linewidth=2, markersize=6)
ax6.plot(results['Z'], results['F_total'], '^-', label='Fisher Information', linewidth=2, markersize=6)
ax6.set_xlabel('Atomic Number Z')
ax6.set_ylabel('Information Measure')
ax6.set_title('(f) All Information Measures')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#Print comparison of relationships
print("\nPaper's reported relationship: S = 6.257 + 1.069 ln(Z)")
print(f"Current fit to paper data: S = {p_log[1]:.3f} + {p_log[0]:.3f} ln(Z)")
print(f"Paper data R² = {r_squared_paper:.4f}")
if len(results['Z']) > 0:
    print(f"\nCurrent calculated relationship: S = {p_calc[1]:.3f} + {p_calc[0]:.3f} ln(Z)")
    print(f"Current data R² = {r_squared_calc:.4f}")
    print(f"\n{'Z':<3} {'Paper S':<10} {'Current S':<10} {'Difference':<12} {'% Error':<10}")
    print("_" * 50)
    for i, z in enumerate(Z_calc):
        if z <= len(S_paper):
            paper_val = S_paper[z-1]  #Paper data starts from Z=1
            our_val = S_calc[i]
            diff = our_val - paper_val
            percent_error = (diff / paper_val) * 100
            print(f"{z:<3} {paper_val:<10.4f} {our_val:<10.4f} {diff:<12.4f} {percent_error:<10.2f}%")

#Try exponential fits for the paper data
ln_S_paper = np.log(S_paper)
#Method 1: S = a * exp(b * Z) - pure exponential form
#Take ln(S) = ln(a) + b*Z, then fit linear
p_exp1 = np.polyfit(Z_paper, ln_S_paper, 1)  #[b, ln(a)]
a_exp1 = np.exp(p_exp1[1])
b_exp1 = p_exp1[0]
S_exp1_fit = a_exp1 * np.exp(b_exp1 * Z_paper)

#Method 2: S = a * Z^b - power law form  
#Take ln(S) = ln(a) + b*ln(Z), then fit linear
ln_Z_paper = np.log(Z_paper)
p_exp2 = np.polyfit(ln_Z_paper, ln_S_paper, 1)  #[b, ln(a)]
a_exp2 = np.exp(p_exp2[1])
b_exp2 = p_exp2[0]
S_exp2_fit = a_exp2 * (Z_paper ** b_exp2)

#Calculate R^2 for both fits
ss_res_exp1 = np.sum((S_paper - S_exp1_fit) ** 2)
ss_tot_exp1 = np.sum((S_paper - np.mean(S_paper)) ** 2)
r_squared_exp1 = 1 - (ss_res_exp1 / ss_tot_exp1)

ss_res_exp2 = np.sum((S_paper - S_exp2_fit) ** 2)
ss_tot_exp2 = np.sum((S_paper - np.mean(S_paper)) ** 2)
r_squared_exp2 = 1 - (ss_res_exp2 / ss_tot_exp2)

#Print exponential fit results
print("Method 1 - Exponential: S = a * exp(b * Z)")
print(f"  S = {a_exp1:.4f} * exp({b_exp1:.4f} * Z)")
print(f"  R² = {r_squared_exp1:.4f}")
print("\nMethod 2 - Power law: S = a * Z^b")
print(f"  S = {a_exp2:.4f} * Z^{b_exp2:.4f}")
print(f"  R² = {r_squared_exp2:.4f}")
print(f"\n  Logarithmic:  R² = {r_squared_paper:.4f}")
print(f"  Exponential:  R² = {r_squared_exp1:.4f}")
print(f"  Power law:    R² = {r_squared_exp2:.4f}")

#Add fits to the existing plot
#Modify the paper data plot to include exponential fits
ax2.plot(Z_paper, S_exp1_fit, 'r:', linewidth=2, 
         label=f'Exp Fit: S = {a_exp1:.3f} * exp({b_exp1:.4f} * Z)')
ax2.plot(Z_paper, S_exp2_fit, 'g:', linewidth=2, 
         label=f'Power Fit: S = {a_exp2:.3f} * Z^{b_exp2:.3f}')
ax2.legend(fontsize=8)

#Create a separate detailed exponential fit plot
fig2 = plt.figure(figsize=(12, 8))

#Fits comparison
ax_exp = plt.subplot(2, 2, 1)
ax_exp.plot(Z_paper, S_paper, 'ko', markersize=4, label='Paper Data')
ax_exp.plot(Z_paper, S_logfit, 'b--', linewidth=2, 
           label=f'Log: S = {p_log[1]:.3f} + {p_log[0]:.3f} ln(Z)')
ax_exp.plot(Z_paper, S_exp1_fit, 'r:', linewidth=2, 
           label=f'Exp: S = {a_exp1:.3f} * exp({b_exp1:.4f} * Z)')
ax_exp.plot(Z_paper, S_exp2_fit, 'g:', linewidth=2, 
           label=f'Power: S = {a_exp2:.3f} * Z^{b_exp2:.3f}')
ax_exp.set_xlabel('Atomic Number Z')
ax_exp.set_ylabel('Shannon Entropy S')
ax_exp.set_title('Comparison of Fitting Methods')
ax_exp.legend(fontsize=9)
ax_exp.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#Print conclusion about best fit
print("The best exponential relationship found is:")
if r_squared_exp2 > r_squared_exp1:
    print(f"Power law form: S = {a_exp2:.4f} * Z^{b_exp2:.4f}")
    print(f"with R² = {r_squared_exp2:.4f}")
else:
    print(f"Pure exponential form: S = {a_exp1:.4f} * exp({b_exp1:.4f} * Z)")
    print(f"with R² = {r_squared_exp1:.4f}")
print("\nHowever, the logarithmic relationship from the paper:")
print(f"S = {p_log[1]:.3f} + {p_log[0]:.3f} ln(Z) with R² = {r_squared_paper:.4f}")
print("appears to provide the best fit to the data.\n")

# Get atomic numbers (Z) for which we have results
Z_values = results['Z']

# Match each Z with its ionization potential (from a predefined dictionary)
IP_values = [ionization_potentials[z] for z in Z_values]

# Total Shannon entropy values (position + momentum) for each atom
S_values = results['S']

# Prepare the plot area – 2 rows, 3 columns of subplots
fig3 = plt.figure(figsize=(14, 10))

# Total Shannon entropy vs Ionization potential 
ax1 = plt.subplot(2, 3, 1)
ax1.plot(IP_values, S_values, 'ro-', linewidth=2, markersize=8)

# Add atomic number labels to each point
for i, z in enumerate(Z_values):
    ax1.annotate(f'Z={z}', (IP_values[i], S_values[i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

ax1.set_xlabel('Ionization Potential (eV)')
ax1.set_ylabel('Shannon Entropy S')
ax1.set_title('Shannon Entropy vs Ionization Potential')
ax1.grid(True, alpha=0.3)

# Compute and display correlation coefficient between S and IP
correlation_coeff = np.corrcoef(IP_values, S_values)[0, 1]
ax1.text(0.05, 0.95, f'Correlation: {correlation_coeff:.3f}',
         transform=ax1.transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8))

# Sr and Sk separately vs IP 
ax2 = plt.subplot(2, 3, 2)
ax2.plot(IP_values, results['Sr'], 'bo-', linewidth=2, markersize=6, label='Sr (position)')
ax2.plot(IP_values, results['Sk'], 'go-', linewidth=2, markersize=6, label='Sk (momentum)')

ax2.set_xlabel('Ionization Potential (eV)')
ax2.set_ylabel('Shannon Entropy Components')
ax2.set_title('Shannon Entropy Components vs IP')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Correlation for each component separately
corr_Sr = np.corrcoef(IP_values, results['Sr'])[0, 1]
corr_Sk = np.corrcoef(IP_values, results['Sk'])[0, 1]
ax2.text(0.05, 0.95, f'Sr corr: {corr_Sr:.3f}\nSk corr: {corr_Sk:.3f}',
         transform=ax2.transAxes, fontsize=9,
         bbox=dict(facecolor='white', alpha=0.8))

# Onicescu energy vs IP
ax3 = plt.subplot(2, 3, 3)
ax3.plot(IP_values, results['O_total'], 'mo-', linewidth=2, markersize=8)

for i, z in enumerate(Z_values):
    ax3.annotate(f'Z={z}', (IP_values[i], results['O_total'][i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

ax3.set_xlabel('Ionization Potential (eV)')
ax3.set_ylabel('Onicescu Energy')
ax3.set_title('Onicescu Energy vs Ionization Potential')
ax3.grid(True, alpha=0.3)

corr_O = np.corrcoef(IP_values, results['O_total'])[0, 1]
ax3.text(0.05, 0.95, f'Correlation: {corr_O:.3f}',
         transform=ax3.transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8))

# Fisher information vs IP
ax4 = plt.subplot(2, 3, 4)
ax4.plot(IP_values, results['F_total'], 'co-', linewidth=2, markersize=8)

for i, z in enumerate(Z_values):
    ax4.annotate(f'Z={z}', (IP_values[i], results['F_total'][i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.set_xlabel('Ionization Potential (eV)')
ax4.set_ylabel('Fisher Information')
ax4.set_title('Fisher Information vs Ionization Potential')
ax4.grid(True, alpha=0.3)

corr_F = np.corrcoef(IP_values, results['F_total'])[0, 1]
ax4.text(0.05, 0.95, f'Correlation: {corr_F:.3f}',
         transform=ax4.transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8))

# Overview of all information measures vs Z (not IP)
ax5 = plt.subplot(2, 3, 5)
ax5.plot(Z_values, S_values, 'ro-', linewidth=2, markersize=6, label='Shannon Entropy')
ax5.plot(Z_values, results['O_total'], 'mo-', linewidth=2, markersize=6, label='Onicescu Energy')
ax5.plot(Z_values, results['F_total'], 'co-', linewidth=2, markersize=6, label='Fisher Information')

ax5.set_xlabel('Atomic Number Z')
ax5.set_ylabel('Information Measures')
ax5.set_title('Information Measures vs Atomic Number')
ax5.legend()
ax5.grid(True, alpha=0.3)

# IP vs Z directly
ax6 = plt.subplot(2, 3, 6)
ax6.plot(Z_values, IP_values, 'ko-', linewidth=2, markersize=8)

for i, z in enumerate(Z_values):
    ax6.annotate(f'{z}', (Z_values[i], IP_values[i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

ax6.set_xlabel('Atomic Number Z')
ax6.set_ylabel('Ionization Potential (eV)')
ax6.set_title('Ionization Potential vs Atomic Number')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print numerical table with all data for cross-check
print("\nCORRELATION ANALYSIS WITH IONIZATION POTENTIAL")
print("_" * 60)
print(f"{'Element':<8} {'Z':<3} {'IP (eV)':<8} {'S':<8} {'Sr':<8} {'Sk':<8} {'O':<8} {'F':<8}")
print("-" * 60)

# Mapping atomic number to symbol – adjust this as needed if more elements included
element_names = ['', '', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']

# Print each element’s data in a row
for i, z in enumerate(Z_values):
    print(f"{element_names[z]:<8} {z:<3} {IP_values[i]:<8.3f} {S_values[i]:<8.4f} "
          f"{results['Sr'][i]:<8.4f} {results['Sk'][i]:<8.4f} {results['O_total'][i]:<8.4f} "
          f"{results['F_total'][i]:<8.4f}")

# Print correlation coefficients
print("\nCorrelation coefficients with ionization potential:")
print(f"Shannon Entropy (total):     {correlation_coeff:.4f}")
print(f"Shannon Entropy (position):  {corr_Sr:.4f}")
print(f"Shannon Entropy (momentum):  {corr_Sk:.4f}")
print(f"Onicescu Energy:             {corr_O:.4f}")
print(f"Fisher Information:          {corr_F:.4f}")

# Interpretation of correlation strength
print("\nInterpretation:")
if abs(correlation_coeff) > 0.7:
    strength = "strong"
elif abs(correlation_coeff) > 0.3:
    strength = "moderate"
else:
    strength = "weak"

direction = "positive" if correlation_coeff > 0 else "negative"
print(f"The Shannon entropy shows a {strength} {direction} correlation with ionization potential.")

# Try polynomial fits (linear and quadratic) for S vs IP
if len(IP_values) > 2:
    # Linear fit S = a*IP + b
    p_linear = np.polyfit(IP_values, S_values, 1)
    S_linear_fit = np.polyval(p_linear, IP_values)

    # Try quadratic only if we have enough data points
    if len(IP_values) > 3:
        p_quad = np.polyfit(IP_values, S_values, 2)
        S_quad_fit = np.polyval(p_quad, IP_values)

        # Compute R^2 for both fits
        ss_res_linear = np.sum((S_values - S_linear_fit) ** 2)
        ss_tot = np.sum((S_values - np.mean(S_values)) ** 2)
        r_squared_linear = 1 - (ss_res_linear / ss_tot)

        ss_res_quad = np.sum((S_values - S_quad_fit) ** 2)
        r_squared_quad = 1 - (ss_res_quad / ss_tot)

        print(f"\nFitting Shannon entropy vs ionization potential:")
        print(f"Linear fit:    S = {p_linear[1]:.4f} + {p_linear[0]:.4f} * IP    (R² = {r_squared_linear:.4f})")
        print(f"Quadratic fit: S = {p_quad[2]:.4f} + {p_quad[1]:.4f} * IP + {p_quad[0]:.4f} * IP²    (R² = {r_squared_quad:.4f})")