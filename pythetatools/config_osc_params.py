osc_param_name = ["delta", "dm2", "sin223", "sin213", "sin2213", "sindelta"]

osc_param_name_to_xlabel = {"delta": {'both': r"$\delta_{CP}$", 0:r"$\delta_{CP}$", 1:r"$\delta_{CP}$"},
                            "dm2":   {'both': r"$\Delta m^2_{32} \mathrm{ (NO)}/|\Delta m^2_{31}| \mathrm{ (IO)}$ $[\mathrm{eV}^2/\mathrm{c}^4]$", 0: r"$\Delta m^2_{32} \, [\mathrm{eV}^2/\mathrm{c}^4]$", 1:r"$|\Delta m^2_{31}| \, [\mathrm{eV}^2/\mathrm{c}^4]$"},
                            "sin223":{'both': r"$\sin^{2} \theta_{23}$", 0:r"$\sin^{2} \theta_{23}$", 1:r"$\sin^{2} \theta_{23}$"},
                            "sin213":{'both': r"$\sin^{2} \theta_{13}$", 0:r"$\sin^{2} \theta_{13}$", 1:r"$\sin^{2} \theta_{13}$"},
                            "sin2213":{'both': r"$\sin^{2} 2\theta_{13}$", 0:r"$\sin^{2} 2\theta_{13}$", 1:r"$\sin^{2} 2\theta_{13}$"},
                            "sindelta": {'both': r"$\sin \delta_{CP}$", 0:r"$\sin \delta_{CP}$", 1:r"$\sin \delta_{CP}$"}}

osc_param_to_title = {"delta":[r"$\delta_{CP}$", r"$\delta_{CP}$"], "dm2":[r"$\Delta m^2_{32}$", r"$|\Delta m^2_{31}|$"], "sin223":[r"$\sin^{2} \theta_{23}$", r"$\sin^{2} \theta_{23}$"], "sin213":[r"$\sin^{2} \theta_{13}$", r"$\sin^{2} \theta_{13}$"], "sin2213":[r"$\sin^{2} 2\theta_{13}$", r"$\sin^{2} 2\theta_{13}$"], "sindelta": [r"$\sin \delta_{CP}$", r"$\sin \delta_{CP}$"]}

osc_param_unit = {"delta":"", "dm2": "$[eV^2/c^4]$", "sin223":"", "sin213":""} #is it used?