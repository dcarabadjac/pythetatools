#For FC files reading
delta_values_FC = ['0.0', '0.392699081699', '-0.785398163397', '0.785398163397',
                '-1.57079632679', '1.57079632679', '-2.35619449019', '2.35619449019',
                '2.59181393921', '3.14159265359' ]
sin223_values_FC = ['0.42', '0.44', '0.46', '0.48', '0.5', '0.52', '0.54', '0.56', '0.58', '0.6', '0.62']
sindelta_values_FC = ['0']
param_values_FC = {'delta':delta_values_FC, 'sin223':sin223_values_FC, 'sindelta':sindelta_values_FC}

true_delta_grid_sorted = [-3.14159265359]+ list(map(float, delta_values_FC))
true_delta_grid_sorted.sort()

true_sin223_grid_sorted = list(map(float, sin223_values_FC))
true_sin223_grid_sorted.sort
true_param_grid_sorted = {'delta':true_delta_grid_sorted, 'sin223':true_sin223_grid_sorted, 'sindelta':sindelta_values_FC}