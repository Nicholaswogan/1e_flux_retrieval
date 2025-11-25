import warnings
warnings.filterwarnings('ignore')

import numpy as np
from matplotlib import pyplot as plt
from photochem._clima import rebin_with_errors
from copy import deepcopy
import dill as pickle
import corner
import colorsys
import matplotlib.colors as mcolors
from tqdm import tqdm
from scipy import constants as const
import gridutils

import retrieval_run
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

def lighten_color(color, amount=0.5):
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        raise ValueError(f"Invalid color format: {color}")

    h, l, s = colorsys.rgb_to_hls(*c)
    l = l + (1 - l) * amount  # Increase lightness
    return colorsys.hls_to_rgb(h, l, s)

def interp_atm_to_pressure(atm, key, P):
    if key == 'pressure':
        val = P
    elif key == 'temperature':
        val = np.interp(np.log10(P),np.log10(atm['pressure'][::-1]), atm[key][::-1])
    else:
        val = 10.0**np.interp(np.log10(P),np.log10(atm['pressure'][::-1]), np.log10(atm[key][::-1]))
    return val

def format_constraint(arr, con_type, precision=1):
    
    if con_type == 'normal':
        lo, med, hi = np.quantile(arr, [0.5-0.68/2, 0.5, 0.5+0.68/2])
        lower_err = med - lo
        upper_err = hi - med
        result_str = f"(${med:.{precision}f}_{{-{lower_err:.{precision}f}}}^{{+{upper_err:.{precision}f}}}$)"
    elif con_type == 'upper':
        hi = np.quantile(arr, 0.95)
        result_str = f"($<{hi:.{precision}f}$)"
    else:
        result_str = ""

    return result_str

def format_number(number, precision=0):
    if number == 0:
        return '0'
    if number > 0:
        sign = ''
    else:
        sign = '-'
    return '$'+(sign+('10^{%.'+str(precision)+'f}')%(np.log10(np.abs(number))))+'$'

def make_case(key):

    _, res = make_case_simple(key)
    
    case = retrieval_run.RETRIEVAL_CASES[key]
    with open('pymultinest/'+key+'/'+key+'.pkl','rb') as f:
        result = pickle.load(f)
    samples = result['samples']
    
    # atms
    atms = []
    for i in tqdm(range(samples.shape[0])):
        x = samples[i,:]
        y = retrieval_run.build_x(x, case['params'])
        atms.append(retrieval_run.make_atm(y[:-2], case['P_interp'], case['T_interp'], case['f_interp']))
    res['atms'] = atms

    atm_at_transit = {}
    atm_at_surface = {}
    for key in atms[0]:
        tmp = []
        tmp1 = []
        for atm in atms:
            tmp.append(interp_atm_to_pressure(atm, key, 1e-3*1e6))
            tmp1.append(atm[key][0])
        atm_at_transit[key] = np.array(tmp)
        atm_at_surface[key] = np.array(tmp1)
    res['atm_at_transit'] = atm_at_transit
    res['atm_at_surface'] = atm_at_surface
    

    # Priors
    np.random.seed(0)
    samples = retrieval_run.sample_prior(case['params'], case['prior_ranges'], samples.shape[0])

    atms = []
    for i in tqdm(range(samples.shape[0])):
        x = samples[i,:]
        y = retrieval_run.build_x(x, case['params'])
        atms.append(retrieval_run.make_atm(y[:-2], case['P_interp'], case['T_interp'], case['f_interp']))
    res['prior_atms'] = atms

    atm_at_transit = {}
    atm_at_surface = {}
    for key in atms[0]:
        tmp = []
        tmp1 = []
        for atm in atms:
            tmp.append(interp_atm_to_pressure(atm, key, 1e-3*1e6))
            tmp1.append(atm[key][0])
        atm_at_transit[key] = np.array(tmp)
        atm_at_surface[key] = np.array(tmp1)
    res['prior_atm_at_transit'] = atm_at_transit
    res['prior_atm_at_surface'] = atm_at_surface

    return case, res

def make_case_simple(key):
    case = retrieval_run.RETRIEVAL_CASES[key]
    with open('pymultinest/'+key+'/'+key+'.pkl','rb') as f:
        result = pickle.load(f)
    samples = result['samples']
    
    res = {}
    # Fluxes
    species = ['CH4','CO','CO2','O2','H2']
    fluxes = {}
    for j,sp in enumerate(species):
        tmp = np.empty(samples.shape[0])
        for i in range(samples.shape[0]):
            x = samples[i,:]
            y = retrieval_run.build_x(x, case['params'])
            tmp[i] = case['flux_interp'][sp](y[:-2])
        fluxes[sp] = tmp
    res['fluxes'] = fluxes

    # CO flux
    tmp = np.empty(samples.shape[0])
    for i in range(samples.shape[0]):
        x = samples[i,:]
        y = retrieval_run.build_x(x, case['params'])
        tmp[i] = retrieval_run.flux_from_vdep(y[:-2], 'CO', 1.2e-4, case['P_interp'], case['T_interp'], case['f_interp'])
    res['CO_flux_down_min'] = -tmp
    
    # Priors on fluxes
    np.random.seed(0)
    samples = retrieval_run.sample_prior(case['params'], case['prior_ranges'], 10000)
    species = ['CH4','CO','CO2','O2','H2']
    prior_fluxes = {}
    for j,sp in enumerate(species):
        tmp = np.empty(samples.shape[0])
        for i in range(samples.shape[0]):
            x = samples[i,:]
            y = retrieval_run.build_x(x, case['params'])
            tmp[i] = case['flux_interp'][sp](y[:-2])
        prior_fluxes[sp] = tmp
    res['prior_fluxes'] = prior_fluxes

    tmp = np.empty(samples.shape[0])
    for i in range(samples.shape[0]):
        x = samples[i,:]
        y = retrieval_run.build_x(x, case['params'])
        tmp[i] = retrieval_run.flux_from_vdep(y[:-2], 'CO', 1.2e-4, case['P_interp'], case['T_interp'], case['f_interp'])
    res['prior_CO_flux_down_min'] = -tmp
    
    return case, res

def nominal_archean_spectra():

    case = retrieval_run.RETRIEVAL_CASES['archean']
    truth = case['data_dict']['truth']

    all_species = ['CH4', 'CO', 'CO2', 'H2O', 'O3','O2']
    excude_mols = {'all':[],'clouds': all_species, 'none': all_species}
    for sp in all_species:
        tmp = deepcopy(all_species)
        tmp.remove(sp)
        excude_mols[sp] = tmp

    wavl = case['data_dict']['wavl']
    spectra = {'wavl': wavl}
    for exclude in excude_mols:
        x = deepcopy(truth)
        if exclude in all_species or exclude == 'none':
            # log10Ptop_cld_copy = 1.99
            x[-2] = 2
        rprs2 = case['model'](x, wavl, atmosphere_kwargs={'exclude_mol': excude_mols[exclude]})
        spectra[exclude] = rprs2

    return spectra

def nominal_archean_plot():

    case = retrieval_run.RETRIEVAL_CASES['archean']
    with open('plotting_data.pkl','rb') as f:
        spectra = pickle.load(f)['spectra']

    plt.rcParams.update({'font.size': 13})
    fig = plt.figure(constrained_layout=False,figsize=[10,4])
    fig.patch.set_facecolor("w")

    gs = fig.add_gridspec(100, 100)

    ax1 = fig.add_subplot(gs[:, :35])
    ax2 = fig.add_subplot(gs[:, 47:])
    axs = [ax1, ax2]

    ax = axs[0]

    y = case['data_dict']['truth']
    # y[1] = 0
    atm = retrieval_run.make_atm(y[:-2], case['P_interp'], case['T_interp'], case['f_interp'])

    species = ['H2O','CO2','N2','O2','CO','H2','CH4','O3']
    colors = ['C0','C2','C6','C5','C4','C3','C1','k']
    for i,sp in enumerate(species):
        ax.plot(atm[sp], atm['pressure']/1e6, lw=2, label=sp, c=colors[i])

    ax.text(0.02, 0.02, 'Archean\nEarth-like\nTRAPPIST-1e\n(True atmos.)', size=11, ha='left', va='bottom', color='k', transform=ax.transAxes)

    ax.text(2.0e-6, 1.5e-3, 'H$_2$O', size=13, ha='left', va='bottom', color='C0')
    ax.text(2.5e-3, 3e-2, 'CO$_2$', size=13, ha='left', va='bottom', color='C2')
    ax.text(1.3e-1, 3e-2, 'N$_2$', size=13, ha='left', va='bottom', color='C6')
    ax.text(1.1e-3, 1e-6, 'O$_2$', size=13, ha='left', va='bottom', color='C5')
    ax.text(6e-2, 1e-6, 'CO', size=13, ha='left', va='bottom', color='C4')
    ax.text(3e-6, 1e-6, 'H$_2$', size=13, ha='left', va='bottom', color='C3')
    ax.text(3e-4, 5e-3, 'CH$_4$', size=13, ha='left', va='bottom', color='C1')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(atm['pressure'][0]/1e6,1e-7)
    ax.set_xlim(1e-9,2)
    ax.set_xticks(10.0**np.arange(-8,1,2))
    ax.set_xlabel('Volume Mixing Ratio')
    ax.set_ylabel('Pressure (bar)')
    ax.grid(alpha=0.4)

    ax1 = ax.twiny()
    ax1.plot(atm['temperature'], atm['pressure']/1e6,c='k', lw=2,ls='--')
    ax1.set_xlim(160,320)
    ax1.set_xlabel('Temperature (K)')

    ax1.text(167, 1e-4, 'Temp.', size=13, ha='left', va='bottom', color='k')

    ax = axs[1]

    # spectra = spectra2
    wavl = spectra['wavl']
    rprs2_1 = spectra['all']
    wv = (wavl[1:] + wavl[:-1])/2
    ax.plot(wv, rprs2_1*1e6, lw=1, c='C3')

    offset = -80
    species = ['CH4','CO2','CO','H2O','O3']
    labels = ['CH$_4$','CO$_2$','CO','H$_2$O','O$_3$']
    colors = ['C1','C2','C4','C0','C6']
    for i,sp in enumerate(species):
        rprs2_2 = spectra[sp]
        # ax.fill_between(wavl[1:], rprs2_1*1e6, rprs2_2*1e6, step='pre', label=labels[i], alpha=0.3,fc=colors[i])
        ax.plot(wv, rprs2_2*1e6+offset, label=labels[i],c=colors[i],zorder=0)
    rprs2_2 = spectra['clouds']
    # ax.fill_between(wavl[1:], rprs2_1*1e6, rprs2_2*1e6, step='pre', label='clouds', alpha=0.3,facecolor='0.5')
    ax.plot(wv, rprs2_2*1e6+offset, label='Clouds',c='C5',zorder=0)

    rprs2_2 = spectra['none']
    ax.plot(wv, rprs2_2*1e6+offset, label='Rayleigh & CIA',c='0.3',zorder=0)

    wavl = np.arange(case['data_dict']['wavl'][0],case['data_dict']['wavl'][-1],0.15)
    rprs2, err = rebin_with_errors(case['data_dict']['wavl'].copy(), case['data_dict']['rprs2'].copy(), case['data_dict']['err'].copy(), wavl)
    wv = (wavl[1:] + wavl[:-1])/2
    wv_err = (wavl[1:] - wavl[:-1])/2
    ax.errorbar(wv, rprs2*1e6, yerr=err*1e6, xerr=wv_err, elinewidth=0.7, marker='o', ms=3, capsize=2, ls='', c='k',
                label='',zorder=500)

    ax.legend(ncol=10,bbox_to_anchor=(0.5, -0.02), loc='lower center',fontsize=10.5, labelcolor='linecolor', handlelength=0, handletextpad=-1.2, frameon=False)
    ax.set_xlim(np.min(wavl),np.max(wavl))
    ax.set_ylim(4895, 5187)
    ax.set_ylabel('Transit Depth (ppm)')
    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.grid(alpha=0.4)

    ax1 = ax.twiny()
    ax1.set_xticks([])
    ax1.errorbar([1], [-1], yerr=[1e-10], xerr=[1], elinewidth=0.7, marker='o', ms=3, capsize=2, ls='', c='k',
                label='Synthetic JWST NIRSpec Prism\n(10 transits)',zorder=500)
    ax1.plot([],[],c='C3',lw=1,label='Archean Earth-like TRAPPIST-1e (True spectrum)')
    ax1.legend(ncol=1,bbox_to_anchor=(0.02, 1.02), loc='upper left',fontsize=10, frameon=False)

    plt.savefig('figures/true_atmosphere_and_spectrum.pdf',bbox_inches='tight')
    # plt.show()

def archean_corner_plot():
    key = 'archean'
    case = retrieval_run.RETRIEVAL_CASES[key]
    with open('pymultinest/'+key+'/'+key+'.pkl','rb') as f:
        result = pickle.load(f)

    param_names = [
        r'$\log_{10}P_\mathrm{CO_2}$',           
        r'$\log_{10}P_\mathrm{O_2}$', 
        r'$\log_{10}P_\mathrm{CO}$', 
        r'$\log_{10}P_\mathrm{H_2}$',
        r'$\log_{10}P_\mathrm{CH_4}$',
        r'$\log_{10}P_\mathrm{cloud}$',
        'offset',
    ]
    plt.rcParams.update({'font.size': 14})
    samples = result['samples']
    samples[:,-1] *= 1e6
    corner.corner(
        data=samples, 
        labels=param_names,
        quantiles=[0.15866, 0.5, 0.8413],
        show_titles=True,
        title_kwargs={'fontsize':12.5},
        truths=case['data_dict']['truth']
    )
    plt.savefig('figures/archean_corner.pdf',bbox_inches='tight')

def archean_corner_flux_plot():
    key = 'archean'
    case = retrieval_run.RETRIEVAL_CASES[key]
    with open('plotting_data.pkl','rb') as f:
        res = pickle.load(f)['res_archean']

    linthresh = 1e4
    symlog_transform = gridutils.symlog_transform_func(linthresh)
    symlog_inverse = gridutils.symlog_inverse_func(linthresh)
    # bins = np.append(-10.0**np.arange(6,16,0.5)[::-1],10.0**np.arange(6,16,0.5))
    # bins = symlog_transform(bins)

    species = ['CO2','O2','CO','H2','CH4']
    vals = []
    vals_prior = []
    truths = []
    for sp in species:
        vals.append(symlog_transform(res['fluxes'][sp]))
        vals_prior.append(symlog_transform(res['prior_fluxes'][sp]))
        val = symlog_transform(case['flux_interp'][sp](case['data_dict']['truth'][:-2]))
        truths.append(val)
    vals = np.array(vals).T
    vals_prior = np.array(vals_prior).T

    param_names = [
        '$\Phi_\mathrm{CO_2}$',           
        '$\Phi_\mathrm{O_2}$', 
        '$\Phi_\mathrm{CO}$', 
        '$\Phi_\mathrm{H_2}$',
        '$\Phi_\mathrm{CH_4}$'
    ]

    max_len = np.max([len(a) for a in [vals, vals_prior]])

    plt.rcParams.update({'font.size': 12})
    fig = corner.corner(
        vals_prior,
        weights=np.ones(len(vals_prior))*(max_len/len(vals_prior)),
        labels=param_names,
        # title_kwargs={'fontsize':12.5},
        truths=truths,
        color=lighten_color('C3',0.6),
        bins=30
    )

    fig = corner.corner(
        vals,
        weights=np.ones(len(vals))*(max_len/len(vals)),
        labels=param_names,
        # title_kwargs={'fontsize':12.5},
        # truths=truths,
        bins=30,
        fig=fig,
    )

    axs = np.array(fig.get_axes()).reshape((5, 5))
    i = 0
    axs_2d = [
        axs[1+i,0+i], axs[2+i,0+i], axs[3+i,0+i], axs[4+i,0+i],
        axs[2+i,1+i], axs[3+i,1+i], axs[4+i,1+i],
        axs[3+i,2+i], axs[4+i,2+i],
        axs[4+i,3+i],
    ]
    axs_1d = [axs[0+i,0+i], axs[1+i,1+i], axs[2+i,2+i], axs[3+i,3+i], axs[4+i,4+i]]
    for ax in axs_2d:
        ax.set_xlim(symlog_transform(-1e14),symlog_transform(1e14))
        ax.set_ylim(symlog_transform(-1e14),symlog_transform(1e14))
        
    for ax in axs_1d:
        ax.set_xlim(symlog_transform(-1e14),symlog_transform(1e14))

    for i in range(1,5):
        ax = axs[i,0]
        ticks = ax.get_yticks()
        ax.set_yticklabels([format_number(a) for a in symlog_inverse(ticks)])

    for j in range(0,5):
        ax = axs[4,j]
        ticks = ax.get_xticks()
        ax.set_xticklabels([format_number(a) for a in symlog_inverse(ticks)])

    plt.savefig('figures/archean_corner_fluxes.pdf', bbox_inches='tight')
    # plt.show()

def methane_flux_plot():
    plt.rcParams.update({'font.size': 13.5})
    fig,ax = plt.subplots(1,1,figsize=[5,4])

    key = 'archean'
    case = retrieval_run.RETRIEVAL_CASES[key]
    with open('plotting_data.pkl','rb') as f:
        res = pickle.load(f)['res_archean']
    
    minval = 1e1
    bins = np.arange(np.log10(minval),14,.25)

    log10CH4 = np.log10(np.clip(res['fluxes']['CH4'],a_min=minval,a_max=np.inf))
    hist, bin_edges = np.histogram(log10CH4,density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c='k', lw=3, label='Retrieved flux (10 transits)')

    log10CH4 = np.log10(np.clip(res['prior_fluxes']['CH4'],a_min=minval,a_max=np.inf))
    hist, bin_edges = np.histogram(log10CH4,density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c='0.7', lw=3, label='Retrieved flux (10 transits)')


    # ax.hist(log10CH4, alpha=0.1, bins=bins, density=True, fc='k')
    # +'($\log_{10}F_\mathrm{CH_4}='+format_constraint(log10CH4,'normal')[2:]
    ax.annotate('Retrieved\n'+'CH$_4$ surface flux', xy=(10, 0.15), xytext=(10-0.72, 0.15+.15),
                arrowprops=dict(facecolor='k',ec='k', arrowstyle="->", lw=1.5),
                fontsize=13, color='k', va='center', ha='center')

    ax.annotate('Prior', xy=(12.3, 0.27), xytext=(12.3+0.72, 0.27+.15),
                arrowprops=dict(facecolor='0.7',ec='0.7', arrowstyle="->", lw=1.5),
                fontsize=14, color='0.7', va='center', ha='center')

    color = 'k'
    val = np.log10(case['flux_interp']['CH4'](case['data_dict']['truth'][:-2]))
    ax.axvline(val,c=color, lw=3, ls=':', label='True value')
    # ax.text(.58, .9, 'True value', size=12, ha='left', va='bottom', color='C1', transform=ax.transAxes)
    ax.annotate('True value', xy=(val, 0.8), xytext=(val+0.4, 0.8),
                arrowprops=dict(facecolor=color,ec=color, arrowstyle="->", lw=1.5),
                fontsize=11.5, color=color, va='center')

    val = np.log10(3.74e9*10)
    ax.axvline(val,c='C3',label='Abiotic upper limit\n(Thompson+2022)',lw=3, ls='--')
    # ax.text(np.log10(3.74e9*10)-0.3, .5, 'Approx. abiotic\nupper limit', size=12, ha='center', va='center', color='grey',rotation=90)
    ax.annotate('Approx. abiotic\nupper limit', xy=(val-0.1, 0.8), xytext=(val-2.3, 0.8),
                arrowprops=dict(facecolor='C3',ec='C3', arrowstyle="->", lw=1.5),
                fontsize=11.5, color='C3', va='center')

    key = 'fluxes'
    FCO = res[key]['CO'] - res['CO_flux_down_min']
    methane_biosig = (res[key]['CH4'] > 3.74e9*10) & ((res[key]['CH4'] > FCO) | (res[key]['H2'] < 0))
    oxygen_biosig = (res[key]['CH4'] > 3.74e9*10) & (res[key]['O2'] > 3.74e9)
    life = methane_biosig | oxygen_biosig

    prob = np.mean(life)
    ax.text(.5, 1.03, r'$P(\Phi \in B \mid data)$ '+'= %.2f'%(prob), size=13.5, ha='center', va='bottom', color='k', transform=ax.transAxes)

    ax.set_xlim(8,14)
    ax.set_ylabel('Probability density')
    ax.set_xlabel(r'CH$_4$ surface flux ($\log_{10}$molec. cm$^{-2}$ s$^{-1}$)')
    # ticks = np.arange(8,14,1)
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(['$10^{%i}$'%a for a in ticks])
    ax.set_ylim(0,0.9)

    plt.savefig('figures/CH4_flux.pdf',bbox_inches='tight')

    # plt.show()

def methane_flux_muscles_plot():

    case_archean = retrieval_run.RETRIEVAL_CASES['archean']
    with open('plotting_data.pkl','rb') as f:
        out = pickle.load(f)
    res_archean = out['res_archean']

    case_archean_muscles = retrieval_run.RETRIEVAL_CASES['archean_muscles']
    res_archean_muscles = out['res_archean_muscles']

    plt.rcParams.update({'font.size': 13.5})
    fig,axs = plt.subplots(1,2,figsize=[11,4])

    ax = axs[0]

    wv, F = np.loadtxt('inputs/TRAPPIST1e_hazmat.txt',skiprows=1).T
    ind = np.argmin(np.abs(wv-1))
    ax.plot(wv[ind:], F[ind:], label='HAZMAT (Peacock+2019)', c='k')
    # print(stars.energy_in_spectrum(wv[ind:], F[ind:]))

    wv, F = np.loadtxt('inputs/TRAPPIST1e_muscles.txt',skiprows=1).T
    ind = np.argmin(np.abs(wv-1))
    ax.plot(wv[ind:], F[ind:], label='MUSCLES (Wilson+2021)', c='C0')
    # print(stars.energy_in_spectrum(wv[ind:], F[ind:]))

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(50,3000)
    ax.set_ylim(3e-4,1e4)
    ax.set_ylabel('Stellar flux at TRAPPIST-1e\n(mW m$^{-2}$ nm$^{-1}$)')
    ax.set_xlabel('Wavelength (nm)')
    ax.legend(ncol=1,bbox_to_anchor=(1.02, -0.02), loc='lower right',fontsize=10.5, frameon=False)

    ax = axs[1]

    case = case_archean_muscles
    res = res_archean_muscles

    minval = 1e7
    bins = np.arange(np.log10(minval),14,.25)

    color = 'C0'
    CH4 = np.clip(res['fluxes']['CH4'],a_min=minval,a_max=np.inf)
    log10CH4 = np.log10(CH4)
    hist, bin_edges = np.histogram(log10CH4,density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=color, lw=3, label='Retrieved flux (10 transits)')
    # ax.hist(log10CH4, alpha=0.1, bins=bins, density=True, fc='k')
    x, y = 9.4,.63
    ax.annotate('Retrieved\n'+'CH$_4$ surface flux\n(Assumes MUSCLES\nstellar spectrum)', xy=(x,y), xytext=(x-0.55, y+.27),
                arrowprops=dict(facecolor=color,ec=color, arrowstyle="->", lw=1.5),
                fontsize=8.5, color=color, va='center', ha='center')

    key = 'fluxes'
    FCO = res[key]['CO'] - res['CO_flux_down_min']
    methane_biosig = (res[key]['CH4'] > 3.74e9*10) & ((res[key]['CH4'] > FCO) | (res[key]['H2'] < 0))
    oxygen_biosig = (res[key]['CH4'] > 3.74e9*10) & (res[key]['O2'] > 3.74e9)
    life = methane_biosig | oxygen_biosig
    prob = np.mean(life)
    ax.text(.13, .4, '$P(\Phi \in B \mid data)$\n'+'= %.2f'%(prob), size=8.5, ha='center', va='bottom', color='C0', transform=ax.transAxes)

    color = 'k'
    case = case_archean
    res = res_archean

    CH4 = np.clip(res['fluxes']['CH4'],a_min=minval,a_max=np.inf)
    log10CH4 = np.log10(CH4)
    hist, bin_edges = np.histogram(log10CH4,density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=color, lw=3, label='Retrieved flux (10 transits)')
    x, y = 12.1,.23
    ax.annotate('Retrieved\n'+'CH$_4$ surface flux\n(Assumes HAZMAT\nstellar spectrum)', xy=(x,y), xytext=(x+0.5, y+.27),
                arrowprops=dict(facecolor=color,ec=color, arrowstyle="->", lw=1.5),
                fontsize=8.5, color=color, va='center', ha='center')
    # CH4 = np.clip(case['fluxes']['CH4'],a_min=minval,a_max=np.inf)
    # log10CH4 = np.log10(CH4)
    # hist, bin_edges = np.histogram(log10CH4,density=True, bins=bins)
    # ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c='k', lw=3, label='Retrieved flux (10 transits)')
    # ax.hist(log10CH4, alpha=0.1, bins=bins, density=True, fc='k')

    key = 'fluxes'
    FCO = res[key]['CO'] - res['CO_flux_down_min']
    methane_biosig = (res[key]['CH4'] > 3.74e9*10) & ((res[key]['CH4'] > FCO) | (res[key]['H2'] < 0))
    oxygen_biosig = (res[key]['CH4'] > 3.74e9*10) & (res[key]['O2'] > 3.74e9)
    life = methane_biosig | oxygen_biosig
    prob = np.mean(life)
    ax.text(.88, .16, '$P(\Phi \in B \mid data)$\n'+'= %.2f'%(prob), size=8.5, ha='center', va='bottom', color='k', transform=ax.transAxes)


    color = 'k'
    val = np.log10(retrieval_run.RETRIEVAL_CASES['archean']['flux_interp']['CH4'](case['data_dict']['truth'][:-2]))
    ax.axvline(val,c=color, lw=3, ls=':', label='True value\n(based on Hazmat stellar spectrum)')
    ax.text(.81, .9, 'True value\n(Assumes HAZMAT\nstellar spectrum)', size=10.5, ha='center', va='center', color=color, transform=ax.transAxes)
    ax.annotate('', xy=(val, 0.9), xytext=(val+0.4, 0.9),
                arrowprops=dict(facecolor=color,ec=color, arrowstyle="->", lw=1.5),
                fontsize=10, color=color, va='center',ha='center')

    val = np.log10(3.74e9*10)
    ax.axvline(val,c='C3',label='Approx. abiotic\nupper limit',lw=3, ls='--')
    # ax.text(np.log10(3.74e9*10)-0.3, .5, 'Approx. abiotic\nupper limit', size=12, ha='center', va='center', color='grey',rotation=90)
    ax.annotate('Approx. abiotic\nupper limit', xy=(val, 0.75), xytext=(val+1.85, 0.75),
                arrowprops=dict(facecolor='C3',ec='C3', arrowstyle="->", lw=1.5),
                fontsize=11, color='C3', va='center', ha='center')

    ax.set_xlim(8,13.5)
    ax.set_ylabel('Probability density')
    ax.set_xlabel('CH$_4$ surface flux ($\log_{10}$molec. cm$^{-2}$ s$^{-1}$)')
    # ticks = np.arange(8,14,1)
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(['$10^{%i}$'%a for a in ticks])
    ax.set_ylim(0,1)

    plt.savefig('figures/CH4_flux_muscles.pdf',bbox_inches='tight')

def abundances_plot():

    case_archean = retrieval_run.RETRIEVAL_CASES['archean']
    with open('plotting_data.pkl','rb') as f:
        out = pickle.load(f)
    res_archean = out['res_archean']

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(constrained_layout=False,figsize=[12,6])
    fig.patch.set_facecolor("w")

    case = case_archean
    res = res_archean

    atm_at_transit = res['atm_at_transit']
    atm_at_surface = res['atm_at_surface']
    fluxes = res['fluxes']
    prior_atm_at_transit = res['prior_atm_at_transit']
    prior_atm_at_surface = res['prior_atm_at_surface']

    den = atm_at_surface['pressure']/(const.k*1e7*atm_at_surface['temperature'])
    CO_max_abiotic_deposition_flux = - atm_at_surface['CO']*den*1e-8
    CO_max_abiotic_flux = fluxes['CO'] - CO_max_abiotic_deposition_flux

    # mask = (fluxes['CH4'] > 3.74e9*10) & (CO_max_abiotic_flux > fluxes['CH4']) & (fluxes['H2'] > 3.74e9) & (fluxes['O2'] < 3.74e9)
    # inds = np.where(mask)
    mask = (fluxes['CH4'] < 1e200)
    inds = np.where(mask)



    gs = fig.add_gridspec(100, 100)

    sep = 3
    w = int((100-(2*sep))/3)
    axs1 = []
    start = 0
    end = w
    for i in range(3):
        ax = fig.add_subplot(gs[:43, start:end])
        axs1.append(ax)
        start = end+sep
        end = end+sep+w


    sep = 2
    w = int((100-(3*sep))/4)
    axs2 = []

    start = 0
    end = w
    for i in range(4):
        ax = fig.add_subplot(gs[57:, start:end])
        axs2.append(ax)
        start = end+sep
        end = end+sep+w

    y = retrieval_run.build_x(case['data_dict']['truth'], case['params'])
    atm_true = retrieval_run.make_atm(y[:-2], case['P_interp'], case['T_interp'], case['f_interp'])


    # CO2, CH4, CO, H2, O2,  H2O

    # Temperature

    # species = ['H2','O2','H2O']


    sp = 'H2'
    ax = axs2[0]
    color_base = 'C3'
    dark_color = lighten_color(color_base, -0.3)
    light_color = lighten_color(color_base, 0.5)

    bins = np.arange(-9.5,0.01,0.5)
    ax.set_xlim(-9,0)
    ax.set_xticks(np.arange(-9,0,2))
    ax.set_xlabel('$\log_{10}(X_\mathrm{H_2})$')

    hist, bin_edges = np.histogram(np.log10(atm_at_transit[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=3, ls='-', label='$10^{-3}$ bar')
    ax.text(0.1, .8, '$10^{-3}$ bar', size=10, ha='left', va='bottom', color=light_color, transform=ax.transAxes)
    hist, bin_edges = np.histogram(np.log10(atm_at_surface[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=3, ls='-', label='Surface')
    ax.text(0.02, .4, 'Surface', size=10, ha='left', va='bottom', color=dark_color, transform=ax.transAxes)

    hist, bin_edges = np.histogram(np.log10(prior_atm_at_transit[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=0.5, ls='-', label='$10^{-3}$ bar')
    hist, bin_edges = np.histogram(np.log10(prior_atm_at_surface[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=0.5, ls='-', label='Surface')

    ax.axvline(np.log10(interp_atm_to_pressure(atm_true, sp, 1e-3*1e6)), lw=2, c=light_color, ls=':')
    ax.axvline(np.log10(atm_true[sp][0]), lw=2, c=dark_color, ls=':')


    sp = 'O2'
    ax = axs2[1]
    color_base = 'C5'
    dark_color = lighten_color(color_base, -0.3)
    light_color = lighten_color(color_base, 0.5)

    bins = np.arange(-14.5,0.01,0.5)
    ax.set_xlim(-14,0)
    ax.set_xticks(np.arange(-14,0,4))
    ax.set_xlabel('$\log_{10}(X_\mathrm{O_2})$')

    hist, bin_edges = np.histogram(np.log10(atm_at_transit[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=3, ls='-', label='$10^{-3}$ bar')
    ax.text(0.25, .75, '$10^{-3}$\nbar', size=10, ha='left', va='bottom', color=light_color, transform=ax.transAxes)
    hist, bin_edges = np.histogram(np.log10(atm_at_surface[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=3, ls='-', label='Surface')
    ax.text(0.03, .45, 'Surf.', size=10, ha='left', va='bottom', color=dark_color, transform=ax.transAxes)

    hist, bin_edges = np.histogram(np.log10(prior_atm_at_transit[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=0.5, ls='-', label='$10^{-3}$ bar')
    hist, bin_edges = np.histogram(np.log10(prior_atm_at_surface[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=0.5, ls='-', label='Surface')

    ax.axvline(np.log10(interp_atm_to_pressure(atm_true, sp, 1e-3*1e6)), lw=2, c=light_color, ls=':')
    ax.axvline(np.log10(atm_true[sp][0]), lw=2, c=dark_color, ls=':')


    sp = 'H2O'
    ax = axs2[2]
    color_base = 'C0'
    dark_color = lighten_color(color_base, -0.3)
    light_color = lighten_color(color_base, 0.5)

    bins = np.arange(-8.5,0.01,0.5)
    ax.set_xlim(-8,0)
    ax.set_xticks(np.arange(-8,0,2))
    ax.set_xlabel('$\log_{10}(X_\mathrm{H_2O})$')

    hist, bin_edges = np.histogram(np.log10(atm_at_transit[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=3, ls='-', label='$10^{-3}$ bar')
    ax.text(0.13, .6, '$10^{-3}$ bar', size=10, ha='left', va='bottom', color=light_color, transform=ax.transAxes)
    hist, bin_edges = np.histogram(np.log10(atm_at_surface[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=3, ls='-', label='Surface')
    ax.text(0.5, .8, 'Surface', size=10, ha='left', va='bottom', color=dark_color, transform=ax.transAxes)

    hist, bin_edges = np.histogram(np.log10(prior_atm_at_transit[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=0.5, ls='-', label='$10^{-3}$ bar')
    hist, bin_edges = np.histogram(np.log10(prior_atm_at_surface[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=0.5, ls='-', label='Surface')

    ax.axvline(np.log10(interp_atm_to_pressure(atm_true, sp, 1e-3*1e6)), lw=2, c=light_color, ls=':')
    ax.axvline(np.log10(atm_true[sp][0]), lw=2, c=dark_color, ls=':')


    sp = 'temperature'
    ax = axs2[3]
    color_base = 'k'
    dark_color = color_base
    light_color = lighten_color(color_base, 0.6)

    bins = np.arange(100,500,10)
    ax.set_xlim(150,450)
    ax.set_xticks(np.arange(200,450,100))
    ax.set_xlabel('Temperature (K)')

    hist, bin_edges = np.histogram(atm_at_transit[sp][inds],density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=3, ls='-', label='$10^{-3}$ bar')
    ax.text(0.18, .6, '$10^{-3}$ bar', size=10, ha='left', va='bottom', color=light_color, transform=ax.transAxes)
    hist, bin_edges = np.histogram(atm_at_surface[sp][inds],density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=3, ls='-', label='Surface')
    ax.text(0.55, .25, 'Surface', size=10, ha='left', va='bottom', color=dark_color, transform=ax.transAxes)

    hist, bin_edges = np.histogram(prior_atm_at_transit[sp],density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=0.5, ls='-', label='$10^{-3}$ bar')
    hist, bin_edges = np.histogram(prior_atm_at_surface[sp],density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=0.5, ls='-', label='Surface')

    ax.axvline(interp_atm_to_pressure(atm_true, sp, 1e-3*1e6), lw=2, c=light_color, ls=':')
    ax.axvline(atm_true[sp][0], lw=2, c=dark_color, ls=':')


    sp = 'CO2'
    ax = axs1[0]
    color_base = 'C2'
    dark_color = lighten_color(color_base, -0.3)
    light_color = lighten_color(color_base, 0.5)

    bins = np.arange(-5.5,0.01,0.5)
    ax.set_xlim(-5,0)
    ax.set_xticks(np.arange(-5,0.1,1))
    ax.set_xlabel('$\log_{10}(X_\mathrm{CO_2})$')

    hist, bin_edges = np.histogram(np.log10(atm_at_transit[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=3, ls='-', label='$10^{-3}$ bar')


    ax.text(0.37, .35, '$10^{-3}$ bar', size=10, ha='left', va='bottom', color=light_color, transform=ax.transAxes)
    hist, bin_edges = np.histogram(np.log10(atm_at_surface[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=3, ls='-', label='Surface')

    hist, bin_edges = np.histogram(np.log10(prior_atm_at_transit[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=0.5, ls='-', label='$10^{-3}$ bar')
    hist, bin_edges = np.histogram(np.log10(prior_atm_at_surface[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=0.5, ls='-', label='Surface')

    ax.text(0.52, .55, 'Surface', size=10, ha='left', va='bottom', color=dark_color, transform=ax.transAxes)
    ax.axvline(np.log10(interp_atm_to_pressure(atm_true, sp, 1e-3*1e6)), lw=2, c=light_color, ls=':')
    ax.axvline(np.log10(atm_true[sp][0]), lw=2, c=dark_color, ls=':')


    sp = 'CH4'
    ax = axs1[1]
    color_base = 'C1'
    dark_color = lighten_color(color_base, -0.3)
    light_color = lighten_color(color_base, 0.5)

    bins = np.arange(-8.5,0.01,0.5)
    ax.set_xlim(-8,0)
    ax.set_xticks(np.arange(-8,0.1,2))
    ax.set_xlabel('$\log_{10}(X_\mathrm{CH_4})$')

    hist, bin_edges = np.histogram(np.log10(atm_at_transit[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=3, ls='-', label='$10^{-3}$ bar')

    ax.text(0.2, .4, '$10^{-3}$ bar', size=10, ha='left', va='bottom', color=light_color, transform=ax.transAxes)
    hist, bin_edges = np.histogram(np.log10(atm_at_surface[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=3, ls='-', label='Surface')

    hist, bin_edges = np.histogram(np.log10(prior_atm_at_transit[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=0.5, ls='-', label='$10^{-3}$ bar')
    hist, bin_edges = np.histogram(np.log10(prior_atm_at_surface[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=0.5, ls='-', label='Surface')

    ax.text(0.23, .55, 'Surface', size=10, ha='left', va='bottom', color=dark_color, transform=ax.transAxes)
    ax.axvline(np.log10(interp_atm_to_pressure(atm_true, sp, 1e-3*1e6)), lw=2, c=light_color, ls=':')
    ax.axvline(np.log10(atm_true[sp][0]), lw=2, c=dark_color, ls=':')


    sp = 'CO'
    ax = axs1[2]
    color_base = 'C4'
    dark_color = lighten_color(color_base, -0.3)
    light_color = lighten_color(color_base, 0.5)

    bins = np.arange(-10.5,0.01,0.5)
    ax.set_xlim(-10,0)
    ax.set_xticks(np.arange(-10,0.1,2))
    ax.set_xlabel('$\log_{10}(X_\mathrm{CO})$')

    hist, bin_edges = np.histogram(np.log10(atm_at_transit[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=3, ls='-', label='$10^{-3}$ bar')


    ax.text(0.67, .45, '$10^{-3}$ bar', size=10, ha='left', va='bottom', color=light_color, transform=ax.transAxes)
    hist, bin_edges = np.histogram(np.log10(atm_at_surface[sp][inds]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=3, ls='-', label='Surface')

    hist, bin_edges = np.histogram(np.log10(prior_atm_at_transit[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=light_color, lw=0.5, ls='-', label='$10^{-3}$ bar')
    hist, bin_edges = np.histogram(np.log10(prior_atm_at_surface[sp]),density=True, bins=bins)
    ax.plot(bin_edges[1:],hist,drawstyle='steps-pre', c=dark_color, lw=0.5, ls='-', label='Surface')

    ax.text(0.12, .27, 'Surface', size=10, ha='left', va='bottom', color=dark_color, transform=ax.transAxes)
    ax.axvline(np.log10(interp_atm_to_pressure(atm_true, sp, 1e-3*1e6)), lw=2, c=light_color, ls=':')
    ax.axvline(np.log10(atm_true[sp][0]), lw=2, c=dark_color, ls=':')

    for ax in axs2:
        ax.set_ylim(0,ax.get_ylim()[1])
        ax.set_yticks([])

    for ax in axs1:
        ax.set_ylim(0,ax.get_ylim()[1])
        ax.set_yticks([])


    ax = axs2[0]
    ax.set_ylabel('Probability Density')
    ax = axs1[0]
    ax.set_ylabel('Probability Density')

    ax = axs1[1]
    ax = ax.twinx()
    ax.set_yticks([])
    ax.plot([],[],c='k',lw=3, label='Posterior')
    ax.plot([],[],c='k',lw=1, label='Prior')
    ax.plot([],[],c='k',lw=2, ls=':', label='Truth')
    ax.legend(ncol=3,bbox_to_anchor=(0.5, 1.02), loc='lower center',fontsize=15)


    plt.savefig('figures/abundances.pdf',bbox_inches='tight')

    # plt.show()


def plotting_data():

    spectra = nominal_archean_spectra()

    _, res_archean = make_case('archean')
    _, res_archean_muscles = make_case('archean_muscles')

    out = {
        'spectra': spectra,
        'res_archean': res_archean,
        'res_archean_muscles': res_archean_muscles,
    }
    with open('plotting_data.pkl','wb') as f:
        pickle.dump(out, f)

def main():
    # plotting_data()
    # nominal_archean_plot()
    # archean_corner_plot()
    # archean_corner_flux_plot()
    # methane_flux_plot()
    # methane_flux_muscles_plot()
    abundances_plot()

if __name__ == '__main__':
    main()