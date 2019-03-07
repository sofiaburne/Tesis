import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


class CoplanaridadPLOTS:
    
    def __init__(self, figsize = (30,15), lw = 1.5, font_title = 30, font_label = 30, font_leg = 20,
                 ticks_l = 6, ticks_w = 3, grid_alpha = 0.8, labelpad = 110, msize = 8, markers = ['o', 's', '^', '*'],
                 colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']):
        
        self.figsize = figsize
        self.lw = lw
        self.font_title = font_title
        self.font_label = font_label
        self.font_leg = font_leg
        self.ticks_l = ticks_l
        self.ticks_w = ticks_w
        self.grid_alpha = grid_alpha
        self.labelpad = labelpad
        self.colors = colors
        self.msize = msize
        self.markers = markers

    
    
    def hist_norm_boot(self, n, av_n, fignum, title, xticks_nx, xticks_ny, xticks_nz, bins = 60):
    
        self.nB_boot = n
        self.av_nB_boot = av_n
        self.lw = 2
        
        
        plt.figure(fignum, figsize = (25,10))
        titulo = r'$\bf{Método}$ $\bf{bootstrap}$,'
        plt.suptitle(titulo+r' {}'.format(title), fontsize = self.font_title)
        plt.subplots_adjust(top=0.88, bottom=0.10, left=0.1, right=0.95, hspace=0.2, wspace=0.2)
        
        p = plt.subplot(131)
        plt.hist(self.nB_boot[:,0], bins = bins, color = self.colors[0])
        plt.axvline(x = self.av_nB_boot[0], linewidth = self.lw, label = r'$n_x$ medio', color = 'k')
        plt.xlabel(r'$n_x$', fontsize = self.font_label)
        plt.ylabel(r'Número de ocurrencias', fontsize = 30)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.xticks(xticks_nx)
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p2 = plt.subplot(132, sharey = p)
        plt.setp(p2.get_yticklabels(), visible = False)
        plt.hist(self.nB_boot[:,1], bins = bins, color = self.colors[1])
        plt.axvline(x = self.av_nB_boot[1], linewidth = self.lw, label = r'$n_y$ medio', color = 'k')
        plt.xlabel(r'$n_y$', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p2.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.xticks(xticks_ny)
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p3 = plt.subplot(133, sharey = p)
        plt.setp(p3.get_yticklabels(), visible = False)
        plt.hist(self.nB_boot[:,2], bins = bins, color = self.colors[2])
        plt.axvline(x = self.av_nB_boot[2], linewidth = self.lw, label = r'$n_z$ medio', color = 'k')
        plt.xlabel(r'$n_z$', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p3.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.xticks(xticks_nz)
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
    
    
    def hist_theta_boot(self, thetaB, av_thetaB, thetaBuV, av_thetaBuV,
                        thetaBdV, av_thetaBdV, thetaBduV, av_thetaBduV,
                        thetaV, av_thetaV, fignum, bins = 70, xtick_spacing = 1):
    
        self.thetaB_boot = thetaB
        self.av_thetaB_boot = av_thetaB
        self.thetaBuV_boot = thetaBuV
        self.av_thetaBuV_boot = av_thetaBuV
        self.thetaBdV_boot = thetaBdV
        self.av_thetaBdV_boot = av_thetaBdV
        self.thetaBduV_boot = thetaBduV
        self.av_thetaBduV_boot = av_thetaBduV
        self.thetaV_boot = thetaV
        self.av_thetaV_boot = av_thetaV
        
        
        plt.figure(fignum, figsize = self.figsize)
        plt.suptitle(r'Histograma $\theta_{Bun}$ bootstrap', fontsize = self.font_title)
        
        p = plt.subplot(151)
        plt.title(r'$n_1$', fontsize = self.font_label)
        plt.hist(self.thetaB_boot[:,0], bins = bins, color = self.colors[6])
        plt.axvline(x = self.av_thetaB_boot, linewidth = self.lw, label = r'$\theta_{Bn}$ medio', color = self.colors[7] )
        plt.xlabel(r'$\theta_{Bun}$ [grados]', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        p.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p2 = plt.subplot(152, sharey = p)
        plt.setp(p2.get_yticklabels(), visible = False)
        plt.title(r'$n_2$', fontsize = self.font_label)
        plt.hist(self.thetaBuV_boot[:,0], bins = bins, color = self.colors[6])
        plt.axvline(x = self.av_thetaBuV_boot, linewidth = self.lw, label = r'$\theta_{Bn}$ medio', color = self.colors[7] )
        plt.xlabel(r'$\theta_{Bun}$ [grados]', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        p2.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p3 = plt.subplot(153, sharey = p)
        plt.setp(p3.get_yticklabels(), visible = False)
        plt.title(r'$n_3$', fontsize = self.font_label)
        plt.hist(self.thetaBdV_boot[:,0], bins = bins, color = self.colors[6])
        plt.axvline(x = self.av_thetaBdV_boot, linewidth = self.lw, label = r'$\theta_{Bn}$ medio', color = self.colors[7])
        plt.xlabel(r'$\theta_{Bun}$ [grados]', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        p3.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p4 = plt.subplot(154, sharey = p)
        plt.setp(p4.get_yticklabels(), visible = False)
        plt.title(r'$n_4$', fontsize = self.font_label)
        plt.hist(self.thetaBuV_boot[:,0], bins = bins, color = self.colors[6])
        plt.axvline(x = self.av_thetaBduV_boot, linewidth = self.lw, label = r'$\theta_{Bn}$ medio', color = self.colors[7])
        plt.xlabel(r'$\theta_{Bun}$ [grados]', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        p4.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p5 = plt.subplot(155, sharey = p)
        plt.setp(p5.get_yticklabels(), visible = False)
        plt.title(r'$n_5$', fontsize = self.font_label)
        plt.hist(self.thetaV_boot[:,0], bins = bins, color = self.colors[6])
        plt.axvline(x = self.av_thetaV_boot, linewidth = self.lw, label = r'$\theta_{Bn}$ medio', color = self.colors[7])
        plt.xlabel(r'$\theta_{Bun}$ [grados]', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        p5.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
    
    
    
    def campos_variacion_updown(self, Bu, av_Bu, norm_Bu, av_norm_Bu,
                                Bd, av_Bd, norm_Bd, av_norm_Bd,
                                fignum, label_B, yticks_u, yticks_d):
        
        self.Bu_s = Bu
        self.av_Bu_s = av_Bu
        self.norm_Bu_s = norm_Bu
        self.av_norm_Bu_s = av_norm_Bu
        self.Bd_s = Bd
        self.av_Bd_s = av_Bd
        self.norm_Bd_s = norm_Bd
        self.av_norm_Bd_s = av_norm_Bd
        

        
        plt.figure(fignum, figsize = (25,15))
        plt.suptitle(r'$\bf{Variación}$ $\bf{de}$ $\bf{intervalos}$ $\bf{upstream}$ $\bf{y}$ $\bf{downstream}$', fontsize = self.font_title)
        plt.subplots_adjust(top=0.92, bottom=0.10, left=0.1, right=0.95, hspace=0.2, wspace=0.2)
        
        ax1 = plt.subplot(211)
        ax1.plot(self.Bu_s[:,0], linestyle='None', marker = self.markers[0], markersize = self.msize, color = self.colors[0])
        ax1.plot(self.Bu_s[:,1], linestyle='None', marker = self.markers[1], markersize = self.msize, color = self.colors[1])
        ax1.plot(self.Bu_s[:,2], linestyle='None', marker = self.markers[2], markersize = self.msize, color = self.colors[2])
        ax1.plot(self.norm_Bu_s, linestyle='None', marker = self.markers[3], markersize = self.msize, color = self.colors[3])
        
        if label_B == 'B':
            ax1.axhline(y = self.av_Bu_s[0], linewidth = self.lw, label = r'$B_{ux}$ medio', color = self.colors[0])
            ax1.axhline(y = self.av_Bu_s[1], linewidth = self.lw, label = r'$B_{uy}$ medio', color = self.colors[1])
            ax1.axhline(y = self.av_Bu_s[2], linewidth = self.lw, label = r'$B_{uz}$ medio', color = self.colors[2])
            ax1.axhline(y = self.av_norm_Bu_s, linewidth = self.lw, label = r'$|B_{u}|$ medio', color = self.colors[3])
            ax1.set_ylabel(r'$B_u$ [nT]'.format(label_B), fontsize = self.font_label)
        
        elif label_B == 'V':
            ax1.axhline(y = self.av_Bu_s[0], linewidth = self.lw, label = r'$V_{ux}$ medio', color = self.colors[0])
            ax1.axhline(y = self.av_Bu_s[1], linewidth = self.lw, label = r'$V_{uy}$ medio', color = self.colors[1])
            ax1.axhline(y = self.av_Bu_s[2], linewidth = self.lw, label = r'$V_{uz}$ medio', color = self.colors[2])
            ax1.axhline(y = self.av_norm_Bu_s, linewidth = self.lw, label = r'$|V_{u}|$ medio', color = self.colors[3])
            ax1.set_ylabel(r'$V_u$ [km/s]'.format(label_B), fontsize = self.font_label)
            
        ax1.axes.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        ax1.set_yticks(yticks_u)
        #ax1.yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))
        ax1.legend(loc = 0, fontsize = self.font_leg)
        ax1.axes.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        ax2 = plt.subplot(212)
        ax2.plot(self.Bd_s[:,0], linestyle='None', marker = self.markers[0], markersize = self.msize, color = self.colors[0])
        ax2.plot(self.Bd_s[:,1], linestyle='None', marker = self.markers[1], markersize = self.msize, color = self.colors[1])
        ax2.plot(self.Bd_s[:,2], linestyle='None', marker = self.markers[2], markersize = self.msize, color = self.colors[2])
        ax2.plot(self.norm_Bd_s, linestyle='None', marker = self.markers[3], markersize = self.msize, color = self.colors[3])
        
        if label_B == 'B':
            ax2.axhline(y = self.av_Bd_s[0], linewidth = self.lw, label = r'$B_{dx}$ medio', color = self.colors[0])
            ax2.axhline(y = self.av_Bd_s[1], linewidth = self.lw, label = r'$B_{dy}$ medio', color = self.colors[1])
            ax2.axhline(y = self.av_Bd_s[2], linewidth = self.lw, label = r'$B_{dz}$ medio', color = self.colors[2])
            ax2.axhline(y = self.av_norm_Bd_s, linewidth = self.lw, label = r'$|B_{d}|$ medio', color = self.colors[3])
            ax2.set_ylabel(r'$B_d$ [nT]'.format(label_B), fontsize = self.font_label)
        
        elif label_B == 'V':
            ax2.axhline(y = self.av_Bd_s[0], linewidth = self.lw, label = r'$V_{dx}$ medio', color = self.colors[0])
            ax2.axhline(y = self.av_Bd_s[1], linewidth = self.lw, label = r'$V_{dy}$ medio', color = self.colors[1])
            ax2.axhline(y = self.av_Bd_s[2], linewidth = self.lw, label = r'$V_{dz}$ medio', color = self.colors[2])
            ax2.axhline(y = self.av_norm_Bd_s, linewidth = self.lw, label = r'$|V_{d}|$ medio', color = self.colors[3])
            ax2.set_ylabel(r'$V_d$ [km/s]'.format(label_B), fontsize = self.font_label)
        
        ax2.set_xlabel(r'Realizaciones', fontsize = self.font_label)
        ax2.axes.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        ax2.set_yticks(yticks_d)
        #ax2.yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))
        ax2.legend(loc = 0, fontsize = self.font_leg)
        ax2.axes.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
    
    
    
    def norm_variacion_updown(self, n_ud, av_n_ud, n_u, av_n_u, n_d, av_n_d, fignum, title, xticks_ud, xticks_u, xticks_d):
        
        self.nB_s2 = n_ud
        self.av_nB_s2 = av_n_ud
        self.nB_su = n_u
        self.av_nB_su = av_n_u
        self.nB_sd = n_d
        self.av_nB_sd = av_n_d
        
        
        plt.figure(fignum, figsize = (25,15))
        titulo = r'$\bf{Variación}$ $\bf{de}$ $\bf{intervalos}$ $\bf{upstream}$ $\bf{y}$ $\bf{downstream}$'
        plt.suptitle(titulo+r',    {}'.format(title), fontsize = self.font_title)
        plt.subplots_adjust(top=0.85, bottom=0.10, left=0.1, right=0.95, hspace=0.2, wspace=0.2)
        
        p = plt.subplot(131)
        plt.title('upstream y downstream', fontsize = self.font_label)
        plt.plot(self.nB_s2[:,0], linestyle='None', marker = self.markers[0], markersize = self.msize, color = self.colors[0])
        plt.plot(self.nB_s2[:,1], linestyle='None', marker = self.markers[1], markersize = self.msize, color = self.colors[1])
        plt.plot(self.nB_s2[:,2], linestyle='None', marker = self.markers[2], markersize = self.msize, color = self.colors[2])
        plt.axhline(y = self.av_nB_s2[0], linewidth = self.lw, label = r'$n_x$ medio', color = self.colors[0])
        plt.axhline(y = self.av_nB_s2[1], linewidth = self.lw, label = r'$n_y$ medio', color = self.colors[1])
        plt.axhline(y = self.av_nB_s2[2], linewidth = self.lw, label = r'$n_z$ medio', color = self.colors[2])
        plt.xlabel(r'Realizaciones', fontsize = self.font_label)
        plt.xticks(xticks_ud)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p2 = plt.subplot(132, sharey = p)
        plt.setp(p2.get_yticklabels(), visible = False)
        plt.title('upstream', fontsize = self.font_label)
        plt.plot(self.nB_su[:,0], linestyle='None', marker = self.markers[0], markersize = self.msize, color = self.colors[0])
        plt.plot(self.nB_su[:,1], linestyle='None', marker = self.markers[1], markersize = self.msize, color = self.colors[1])
        plt.plot(self.nB_su[:,2], linestyle='None', marker = self.markers[2], markersize = self.msize, color = self.colors[2])
        plt.axhline(y = self.av_nB_su[0], linewidth = self.lw, label = r'$n_x$ medio', color = self.colors[0])
        plt.axhline(y = self.av_nB_su[1], linewidth = self.lw, label = r'$n_y$ medio', color = self.colors[1])
        plt.axhline(y = self.av_nB_su[2], linewidth = self.lw, label = r'$n_z$ medio', color = self.colors[2])
        plt.xlabel(r'Realizaciones', fontsize = self.font_label)
        plt.xticks(xticks_u)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        plt.legend(loc = 5, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p3 = plt.subplot(133, sharey = p)
        plt.setp(p3.get_yticklabels(), visible = False)
        plt.title('downstream', fontsize = self.font_label)
        plt.plot(self.nB_sd[:,0], linestyle='None', marker = self.markers[0], markersize = self.msize, color = self.colors[0])
        plt.plot(self.nB_sd[:,1], linestyle='None', marker = self.markers[1], markersize = self.msize, color = self.colors[1])
        plt.plot(self.nB_sd[:,2], linestyle='None', marker = self.markers[2], markersize = self.msize, color = self.colors[2])
        plt.axhline(y = self.av_nB_sd[0], linewidth = self.lw, label = r'$n_x$ medio', color = self.colors[0])
        plt.axhline(y = self.av_nB_sd[1], linewidth = self.lw, label = r'$n_y$ medio', color = self.colors[1])
        plt.axhline(y = self.av_nB_sd[2], linewidth = self.lw, label = r'$n_z$ medio', color = self.colors[2])
        plt.xlabel(r'Realizaciones', fontsize = self.font_label)
        plt.xticks(xticks_d)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        plt.legend(loc = 5, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
    
    
    
    def theta_variacion_updown(self, theta_ud, av_theta_ud, theta_u, av_theta_u, theta_d, av_theta_d, fignum, title, xticks_ud, xticks_u, xticks_d):
        
        self.thetaB_s2 = theta_ud
        self.av_thetaB_s2 = av_theta_ud
        self.thetaB_su = theta_u
        self.av_thetaB_su = av_theta_u
        self.thetaB_sd = theta_d
        self.av_thetaB_sd = av_theta_d
        
        
        plt.figure(fignum, figsize = (25,15))
        titulo = r'$\bf{Variación}$ $\bf{de}$ $\bf{intervalos}$ $\bf{upstream}$ $\bf{y}$ $\bf{downstream}$'
        plt.suptitle(titulo+r',    {}'.format(title), fontsize = self.font_title)
        plt.subplots_adjust(top=0.85, bottom=0.10, left=0.1, right=0.95, hspace=0.2, wspace=0.2)
        
        p = plt.subplot(131)
        plt.title('upstream y downstream', fontsize = self.font_label)
        plt.plot(self.thetaB_s2, linestyle='None', marker = self.markers[3], markersize = self.msize, color = self.colors[3])
        plt.axhline(y = self.av_thetaB_s2, linewidth = self.lw, label = r'$\theta_{Bn}$ medio', color = self.colors[3])
        plt.ylabel(r'$\theta_{Bn}$ [°]', fontsize = self.font_label)
        plt.xlabel(r'Realizaciones', fontsize = self.font_label)
        plt.xticks(xticks_ud)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p2 = plt.subplot(132, sharey = p)
        plt.setp(p2.get_yticklabels(), visible = False)
        plt.title('upstream', fontsize = self.font_label)
        plt.plot(self.thetaB_su, linestyle='None', marker = self.markers[3], markersize = self.msize, color = self.colors[3])
        plt.axhline(y = self.av_thetaB_su, linewidth = self.lw, label = r'$\theta_{Bn}$ medio', color = self.colors[3])
        plt.xlabel(r'Realizaciones', fontsize = self.font_label)
        plt.xticks(xticks_u)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p3 = plt.subplot(133, sharey = p)
        plt.setp(p3.get_yticklabels(), visible = False)
        plt.title('downstream', fontsize = self.font_label)
        plt.plot(self.thetaB_sd, linestyle='None', marker = self.markers[3], markersize = self.msize, color = self.colors[3])
        plt.axhline(y = self.av_thetaB_sd, linewidth = self.lw, label = r'$\theta_{Bn}$ medio', color = self.colors[3])
        plt.xlabel(r'Realizaciones', fontsize = self.font_label)
        plt.xticks(xticks_d)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        
    
    def hist_campos_variacion_updown(self, B, av_B, norm_B, av_norm_B, fignum, title, label_Bu, label_Bux, label_Buy, label_Buz, bins = 15, xtick_spacing = 0.2):
        
        self.Bu_s = B
        self.av_Bu_s = av_B
        self.norm_Bu_s = norm_B
        self.av_norm_Bu_s = av_norm_B
        self.font_leg = 15
        
        
        plt.figure(fignum, figsize = self.figsize)
        plt.suptitle(r'{}  -  Variación de intervalos upstream/downstream'.format(title), fontsize = self.font_title)
                
        p = plt.subplot(221)
        plt.hist(self.norm_Bu_s, bins = bins, color = self.colors[6])
        plt.axvline(x = self.av_norm_Bu_s, linewidth = self.lw, label = r'|{}| medio'.format(label_Bu), color = self.colors[7])
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.xlabel(r'|{}| [nT]'.format(label_Bu), fontsize = self.font_label)
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p2 = plt.subplot(222, sharey = p)
        plt.setp(p2.get_yticklabels(), visible = False)
        plt.hist(self.Bu_s[:,0], bins = bins, color = self.colors[0])
        plt.axvline(x = self.av_Bu_s[0], linewidth = self.lw, label = r'{} medio'.format(label_Bux), color = self.colors[1])
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p2.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.xlabel(r'{} [nT]'.format(label_Bux), fontsize = self.font_label)
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p3 = plt.subplot(223, sharey = p)
        plt.hist(self.Bu_s[:,1], bins = bins, color = self.colors[2])
        plt.axvline(x = self.av_Bu_s[1], linewidth = self.lw, label = r'{} medio'.format(label_Buy), color = self.colors[3])
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p3.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.xlabel(r'{} [nT]'.format(label_Buy), fontsize = self.font_label)
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p4 = plt.subplot(224, sharey = p)
        plt.setp(p4.get_yticklabels(), visible = False)
        plt.hist(self.Bu_s[:,2], bins = bins, color = self.colors[4])
        plt.axvline(x = self.av_Bu_s[2], linewidth = self.lw, label = r'{} medio'.format(label_Buz), color = self.colors[5])
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p4.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.xlabel(r'{} [nT]'.format(label_Buz), fontsize = self.font_label)
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
    
    
    
    def hist_norm_variacion_updown(self,n_ud, av_n_ud, n_u, av_n_u, n_d, av_n_d,
                                   title, fignum, bins = 15, xtick_spacing = 0.2):
        
        self.nB_s2 = n_ud
        self.av_nB_s2 = av_n_ud
        self.nB_su = n_u
        self.av_nB_su = av_n_u
        self.nB_sd = n_d
        self.av_nB_sd = av_n_d
       
    
        plt.figure(fignum, figsize = self.figsize)
        plt.suptitle(r'{}  -  Variación de intervalos upstream/downstream'.format(title), fontsize = self.font_title)
        
        
        p = plt.subplot(331)
        plt.hist(self.nB_s2[:,0], bins = bins, color = self.colors[0])
        plt.axvline(x = self.av_nB_s2[0], linewidth = self.lw, label = r'$n_x$ medio', color = self.colors[1])
        plt.ylabel('upstream\ndownstream', fontsize = self.font_label, rotation = 'horizontal', labelpad = self.labelpad)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p4 = plt.subplot(334, sharey = p)
        plt.hist(self.nB_su[:,0], bins = bins, color = self.colors[0])
        plt.axvline(x = self.av_nB_su[0], linewidth = self.lw, label = r'$n_x$ medio', color = self.colors[1])    
        plt.ylabel('upstream', fontsize = self.font_label, rotation = 'horizontal', labelpad = self.labelpad)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        p4.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p7 = plt.subplot(337, sharey = p)
        plt.hist(self.nB_sd[:,0], bins = bins, color = self.colors[0])
        plt.axvline(x = self.av_nB_sd[0], linewidth = self.lw, label = r'$n_x$ medio', color = self.colors[1])
        plt.ylabel('downstream', fontsize = self.font_label, rotation = 'horizontal', labelpad = self.labelpad)
        plt.xlabel(r'$n_x$', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p7.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p2 = plt.subplot(332, sharey = p)
        plt.setp(p2.get_yticklabels(), visible = False)
        plt.hist(self.nB_s2[:,1], bins = bins, color = self.colors[2])
        plt.axvline(x = self.av_nB_s2[1], linewidth = self.lw, label = r'$n_y$ medio', color = self.colors[3])
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p2.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p5 = plt.subplot(335, sharey = p)
        plt.setp(p5.get_yticklabels(), visible = False)
        plt.hist(self.nB_su[:,1], bins = bins, color = self.colors[2])
        plt.axvline(x = self.av_nB_su[1], linewidth = self.lw, label = r'$n_y$ medio', color = self.colors[3])
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p5.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p8 = plt.subplot(338, sharey = p)
        plt.setp(p8.get_yticklabels(), visible = False)
        plt.hist(self.nB_sd[:,1], bins = bins, color = self.colors[2])
        plt.axvline(x = self.av_nB_sd[1], linewidth = self.lw, label = r'$n_y$ medio', color = self.colors[3])
        plt.xlabel(r'$n_y$', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p8.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p3 = plt.subplot(333, sharey = p)
        plt.setp(p3.get_yticklabels(), visible = False)
        plt.hist(self.nB_s2[:,2], bins = bins, color = self.colors[4])
        plt.axvline(x = self.av_nB_s2[2], linewidth = self.lw, label = r'$n_z$ medio', color = self.colors[5])
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p3.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p6 = plt.subplot(336, sharey = p)
        plt.setp(p6.get_yticklabels(), visible = False)
        plt.hist(self.nB_su[:,2], bins = bins, color = self.colors[4])
        plt.axvline(x = self.av_nB_su[2], linewidth = self.lw, label = r'$n_z$ medio', color = self.colors[5])
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p6.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p9 = plt.subplot(339, sharey = p)
        plt.setp(p9.get_yticklabels(), visible = False)
        plt.hist(self.nB_sd[:,2], bins = bins, color = self.colors[4])
        plt.axvline(x = self.av_nB_sd[2], linewidth = self.lw, label = r'$n_z$ medio', color = self.colors[5])
        plt.xlabel(r'$n_z$', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p9.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)



    
    def hist_theta_variacion_updown(self, theta_ud, av_theta_up, theta_u, av_theta_u, theta_d, av_theta_d,
                                    title, fignum, bins = 15, xtick_spacing = 5):
        
        self.thetaB_s2 = theta_ud
        self.av_thetaB_s2 = av_theta_up
        self.thetaB_su = theta_u
        self.av_thetaB_su = av_theta_u
        self.thetaB_sd = theta_d
        self.av_thetaB_sd = av_theta_d
        
        
        plt.figure(fignum, figsize = self.figsize)
        plt.suptitle(r'{}  -  Variación de intervalos upstream/downstream'.format(title), fontsize = self.font_title)
        
        p = plt.subplot(131)
        plt.title('upstream - downstream', fontsize = self.font_label)
        plt.hist(self.thetaB_s2, bins = bins, color = self.colors[6])
        plt.axvline(x = self.av_thetaB_s2, linewidth = self.lw, label = r'$\theta_{Bun}$ medio', color = self.colors[7])
        plt.xlabel(r'$\theta_{Bun}$ [grados]', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p2 = plt.subplot(132, sharey = p)
        plt.setp(p2.get_yticklabels(), visible = False)
        plt.title('upstream', fontsize = self.font_label)
        plt.hist(self.thetaB_su, bins = bins, color = self.colors[6])
        plt.axvline(x = self.av_thetaB_su, linewidth = self.lw, label = r'$\theta_{Bun}$ medio', color = self.colors[7])
        plt.xlabel(r'$\theta_{Bun}$ [grados]', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p2.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        
        p3 = plt.subplot(133, sharey = p)
        plt.setp(p3.get_yticklabels(), visible = False)
        plt.title('downstream', fontsize = self.font_label)
        plt.hist(self.thetaB_sd, bins = bins, color = self.colors[6])
        plt.axvline(x = self.av_thetaB_sd, linewidth = self.lw, label = r'$\theta_{Bun}$ medio', color = self.colors[7])
        plt.xlabel(r'$\theta_{Bun}$ [grados]', fontsize = self.font_label)
        plt.tick_params(axis = 'both', which = 'both', length = self.ticks_l, width = self.ticks_w, labelsize = self.font_label)
        #p3.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
        plt.legend(loc = 0, fontsize = self.font_leg)
        plt.grid(which = 'both', axis = 'both', linewidth = self.lw, linestyle = '--', alpha = self.grid_alpha)
        