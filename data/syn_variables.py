het_cols = ['Si','Al','P','Ge','B','V','Be','Ga','Ti'] # heteroatoms in the dataset
gel_cols = ['Si', 'Al', 'P', 'Ge', 'B', 'Na', 'K', 'OH', 'F', 'H2O', 'sda1']
ratios = {'Si/Al': {'quantile': 0.98, 'name': 'Si/Al'}, # gel ratios in dataset. quantiles are used to cap the values at a certain value.
          'Al/P': {'quantile': 0.99, 'name': 'Al/P'},
          'Si/Ge': {'quantile': 0.995, 'name': 'Si/Ge'},
          'Si/B': {'quantile': 0.99, 'name': 'Si/B'},
          'Na/T': {'quantile': 0.95, 'name': 'Na/T'},
          'K/T': {'quantile': 0.99, 'name': 'K/T'},
          'OH/T': {'quantile': 0.97, 'name': 'OH/T'},
          'F/T': {'quantile': 0.95, 'name': 'F/T'},
          'H2O/T': {'quantile': 0.99, 'name': 'H$_2$O/T'},
          'sda1/T': {'quantile': 0.98, 'name': 'SDA/T'},
          }

conds = {'cryst_temp': {'name': 'Cryst. temp. ($^\circ$C)'}, # synthesis conditions in dataset.
         'cryst_time': {'name': 'Cryst. time ($^\circ$C)'},
         }
osda_cols = {
    'asphericity_mean_0': 'OSDA asphericity',
        'axes_mean_0': 'OSDA axis 1',
        'axes_mean_1': 'OSDA axis 2',
        'formal_charge': 'OSDA charge',
        'free_sasa_mean_0': 'OSDA SASA',
        'mol_weight': 'OSDA molecular weight',
        'npr1_mean_0': 'OSDA NPR 1',
        'npr2_mean_0': 'OSDA NPR 2',
        'num_rot_bonds_mean_0': 'OSDA rotatable bonds',
        'pmi1_mean_0': 'OSDA PMI 1',
        'pmi2_mean_0': 'OSDA PMI 2',
        'pmi3_mean_0': 'OSDA PMI 3',
        'spherocity_index_mean_0': 'OSDA spherocity',
        'volume_mean_0': 'OSDA volume',
}   
zeo_cols  = ['zeo_num_atoms', 'zeo_a', 'zeo_b', 'zeo_c', 'zeo_alpha', 'zeo_beta', 'zeo_gamma', 'zeo_volume', 'zeo_largest_free_sphere', 'zeo_largest_free_sphere_a', 'zeo_largest_free_sphere_b', 'zeo_largest_free_sphere_c', 'zeo_largest_included_sphere', 'zeo_largest_included_sphere_a', 'zeo_largest_included_sphere_b', 'zeo_largest_included_sphere_c', 'zeo_largest_included_sphere_fsp', 'zeo_num_atoms_per_vol', 'zeo_chan_num_channels', 'zeo_chan_0_dim', 'zeo_chan_1_dim', 'zeo_chan_2_dim', 'zeo_chan_3_dim', 'zeo_chan_4_dim', 'zeo_chan_5_dim', 'zeo_chan_0_di', 'zeo_chan_0_df', 'zeo_chan_0_dif', 'zeo_chan_1_di', 'zeo_chan_1_df', 'zeo_chan_1_dif', 'zeo_chan_2_di', 'zeo_chan_2_df', 'zeo_chan_2_dif', 'zeo_chan_3_di', 'zeo_chan_3_df', 'zeo_chan_3_dif', 'zeo_chan_4_di', 'zeo_chan_4_df', 'zeo_chan_4_dif', 'zeo_chan_5_di', 'zeo_chan_5_df', 'zeo_chan_5_dif', 'zeo_density', 'zeo_asa_a2', 'zeo_asa_m2_cm3', 'zeo_asa_m2_g', 'zeo_nasa_a2', 'zeo_nasa_m2_cm3', 'zeo_nasa_m2_g', 'zeo_sa_num_channels', 'zeo_chan_0_sa_a2', 'zeo_chan_1_sa_a2', 'zeo_chan_2_sa_a2', 'zeo_chan_3_sa_a2', 'zeo_chan_4_sa_a2', 'zeo_chan_5_sa_a2', 'zeo_sa_num_pockets', 'zeo_poc_0_sa_a2', 'zeo_poc_1_sa_a2', 'zeo_poc_2_sa_a2', 'zeo_poc_3_sa_a2', 'zeo_poc_4_sa_a2', 'zeo_poc_5_sa_a2', 'zeo_poc_6_sa_a2', 'zeo_poc_7_sa_a2', 'zeo_poc_8_sa_a2', 'zeo_poc_9_sa_a2', 'zeo_poc_10_sa_a2', 'zeo_poc_11_sa_a2', 'zeo_poc_12_sa_a2', 'zeo_poc_13_sa_a2', 'zeo_poc_14_sa_a2', 'zeo_poc_15_sa_a2', 'zeo_poc_16_sa_a2', 'zeo_poc_17_sa_a2', 'zeo_poc_18_sa_a2', 'zeo_poc_19_sa_a2', 'zeo_poc_20_sa_a2', 'zeo_poc_21_sa_a2', 'zeo_poc_22_sa_a2', 'zeo_poc_23_sa_a2', 'zeo_poc_24_sa_a2', 'zeo_av_a3', 'zeo_av_frac', 'zeo_av_cm3_g', 'zeo_nav_a3', 'zeo_nav_frac', 'zeo_nav_cm3_g', 'zeo_chan_0_vol_a3', 'zeo_chan_1_vol_a3', 'zeo_chan_2_vol_a3', 'zeo_chan_3_vol_a3', 'zeo_chan_4_vol_a3', 'zeo_chan_5_vol_a3', 'zeo_poc_0_vol_a3', 'zeo_poc_1_vol_a3', 'zeo_poc_2_vol_a3', 'zeo_poc_3_vol_a3', 'zeo_poc_4_vol_a3', 'zeo_poc_5_vol_a3', 'zeo_poc_6_vol_a3', 'zeo_poc_7_vol_a3', 'zeo_poc_8_vol_a3', 'zeo_poc_9_vol_a3', 'zeo_poc_10_vol_a3', 'zeo_poc_11_vol_a3', 'zeo_poc_12_vol_a3', 'zeo_poc_13_vol_a3', 'zeo_poc_14_vol_a3', 'zeo_poc_15_vol_a3', 'zeo_poc_16_vol_a3', 'zeo_poc_17_vol_a3', 'zeo_poc_18_vol_a3', 'zeo_poc_19_vol_a3', 'zeo_poc_20_vol_a3', 'zeo_poc_21_vol_a3', 'zeo_poc_22_vol_a3', 'zeo_poc_23_vol_a3', 'zeo_poc_24_vol_a3', 'zeo_poav_a3', 'zeo_poav_frac', 'zeo_poav_cm3_g', 'zeo_ponav_a3', 'zeo_ponav_frac', 'zeo_ponav_cm3_g', 'zeo_probe_rad', 'zeo_N_points', 'zeo_probe_ctr_A_fract', 'zeo_probe_ctr_NA_fract', 'zeo_A_fract', 'zeo_NA_fract', 'zeo_narrow_fract', 'zeo_ovlpvfract', 'zeo_deriv_mean', 'zeo_deriv_variance', 'zeo_deriv_skewness', 'zeo_deriv_kurtosis', 'zeo_cum_mean', 'zeo_cum_variance', 'zeo_cum_skewness', 'zeo_cum_kurtosis', 'zeo_num_si']
