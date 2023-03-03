from queue import Empty
from builtins import print
from cProfile import label
from turtle import width
import numpy as np
import pandas as pd

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os

import json

from stable_baselines3.common.results_plotter import load_results, ts2xy


# =======================================
# ==== Synthetic psychologist plots =====
# =======================================

# -------------------------------------
# for plotting the learning curve

def moving_average(values, window):
	"""
	Smooth values by doing a moving average
	:param values: (numpy array)
	:param window: (int)
	:return: (numpy array)
	"""
	weights = np.repeat(1.0, window) / window
	return np.convolve(values, weights, 'valid')

# -------------------------------------

def plot_learning_curve(output_dir, name=None):
    x, y = ts2xy(load_results(output_dir), 'timesteps')
    w = 1000
    y = moving_average(y, window=w)
    # Truncate x
    x = x[len(x) - len(y):]
    fig = plt.figure()
    
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps /')
    plt.ylabel('Rewards')
    #plt.title(f'last average (window={w})= {np.round(y[-1],3)}')
    plt.title(f'window = {w}; mean reward = {np.round(np.mean(y),3)}')
    plt.savefig(f'{output_dir}learning_curve-{name}.png')
    plt.close()
    return fig



# -------------------------------------

def plot_vic( pdf, data, hyper, v, i, c, af, l, title  ):
	# a function for plotting results. 
	# Plot the results in filename using values v, index i, cols c and aggregation function af.
	table = pd.pivot_table(data, values=[v], index=[i], columns=[c], aggfunc=af)
	print("")
	print(table)
	table.plot() # try grabbing gcf.
	fig = plt.gcf()
	#plt.title(f'{v} against {i} for levels of {c}')
	plt.title(title)
	plt.ylabel(v)
	plt.legend(labels=data[c].unique(), title=c, loc=l)
	output_dir = hyper['output_dir']
	#plt.savefig(f'{output_dir}model/png/{v}_{af.__name__}_against_{i}_for_{c}_')
	plt.close()
	pdf.savefig(fig)
	return fig


def plot_boiler( pdf, model ):

	def write(txt):
		nonlocal y
		plt.text(x, y, txt, ha='left', va='bottom', size=11)
		y -= 1

	lines = 20
	x = 0
	y = lines

	fig1 = plt.figure()
	plt.axis('off')
	write("Corati")
	write('Model')
	for i in model.hyper:
		write( i + " = " + json.dumps(model.hyper[i]))
#		write( json.dumps(i) )# + json.dumps( model.pars[i]) )
	plt.xlim(0,10)
	plt.ylim(0,lines)
	plt.close()
	pdf.savefig(fig1)

	fig2 = plt.figure()
	y=lines
	plt.axis('off')
	write("Corati")
	write('Model')
	for i in model.pars:
		write( i + " in " + json.dumps(model.pars[i]['bounds']))
#		write( json.dumps(i) )# + json.dumps( model.pars[i]) )
	plt.xlim(0,10)
	plt.ylim(0,lines)
	plt.close()
	pdf.savefig(fig2)

	return [fig1, fig2]

# -------------------------------------



# =======================================
# ========== Inference plots ============
# =======================================
def plot_participant_convergence( estimates, participant ):
    output_dir = participant.hyper['output_dir']
    os.makedirs( output_dir , exist_ok=True ) 

    # extract the ground truth parameters and their bounds
    part_pars = participant.parameters_of_interest
    bounds = [ participant.pars[key]['bounds'] for key in part_pars ]
    ground_truth = { key : participant.pars[key]['value'] for key in part_pars}
    
    # prepare the figure
    fig = plt.figure()
    fig.suptitle('True model: ' + participant.name)

    for i, key in zip(range(len(part_pars)), part_pars):
        ax = fig.add_subplot(1, len(part_pars), i+1)
        ax.set_title( key )

        dif = []
        for estimate in estimates:
            if type(estimate) is not dict:
                dif.append(1.)
                continue

            inf_pars = list(estimate.keys())
            if inf_pars == participant.parameters_of_interest:
                dif.append( np.linalg.norm( (estimate[key] - ground_truth[key]) / (np.max(bounds) - np.min(bounds))) )
            else:
                dif.append( 1. )

        ax.plot([x+1 for x in range(len(estimates))], dif)
    plt.tight_layout()
    plt.savefig(output_dir + 'result.png', dpi=300)
    plt.close()
    return



def plot_parameter_marginals(theta, estimate=None, ground_truth=None, output_dir='/',  model_name=None, it=0, thr=None):
    # TODO: this should also take a model name. Parameters for all models should be plotted here!
    # create directory for saving results
    output_dir = output_dir + 'theta_log/' 
    os.makedirs( output_dir , exist_ok=True ) 

    # identify the parameters that require plots
    frame = dict()
    i = 0
    
    for key in theta.keys():
        if isinstance(theta[key]['value'], pd.Series):
            frame[key] = theta[key]['value']

    keys = list(frame.keys())
    keys.sort()
    fig = plt.figure()
    
    for i, key in zip(range(len(keys)), keys):
        ax = fig.add_subplot(1, len(keys), i+1)

        # set label for x-axis, turn off y-axis
        ax.set_title( key )

        ax.set_xlim(theta[key]['bounds']) # TODO: THIS WORKS ONLY FOR THE MEMORY MODEL
        ax.get_yaxis().set_visible(False)

        # plot marginal distribution
        sns.histplot(x = frame[key], bins=20, ax = ax)
            
        # plot estimates
        if estimate is not None:
            ax.axvline(estimate[key], ls = 'dashed', color="black", label='Estimate')

        # plot ground truth
        if ground_truth is not None:
            ax.axvline(ground_truth[key], ls = '-', color="red", label='Ground Truth')
        i += 1

    if estimate is not None:
        estimate = { key : estimate[key] for key in keys}
        print('Estimate: ', estimate)

        if ground_truth is not None:
            print('Ground Truth: ', ground_truth)
        plt.legend()

    if thr is not None and model_name:
        fig.suptitle(f'Model: {model_name}, marginal likelihood: {thr}')
    elif model_name:
        fig.suptitle(f'Model: {model_name}')
        
    plt.tight_layout()
    plt.savefig(output_dir + 'thetas-marginals-' + str(it) + '.png', dpi=300)
    plt.close()
    return


def plot_model_marginals(model_prior, output_dir='/', it=0):
    output_dir = output_dir + 'model_log/' 
    os.makedirs( output_dir , exist_ok=True ) 

    plt.clf()
    fig = plt.figure()
    keys = model_prior.keys()
    values = model_prior.values()
    plt.bar(keys, values)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir + 'm-marginals-' + str(it) + '.png', dpi=300)
    plt.close()


def plot_correlation(estimates, ground_truths, dists, output_dir, keys, bounds):
    plt.clf()
    fig = plt.figure()
    mean_dist = np.mean([x[-1] for x in dists])
    std_dist = np.std([x[-1] for x in dists])
    fig.suptitle(f'Distance (mean $\pm$ std): {round(mean_dist, 3)} $\pm$ {round(std_dist, 3)}') 

    for i in range(len(keys)):
        ax = fig.add_subplot(1, len(keys), i+1)

        cur_key = keys[i]
        cur_bounds = bounds[i]

        ax.set_xlim(cur_bounds)
        ax.set_ylim(cur_bounds)

        ax.set_title( cur_key )    
        ax.scatter(np.asarray(estimates)[:,i], np.asarray(ground_truths)[:,i])
        ax.set_xlabel('Estimate')

        if i == 0:
            ax.set_ylabel('Ground truth')
        else:
            ax.set_ylabel('')
    
    plt.tight_layout()        
    plt.savefig(output_dir + 'result.png', dpi=300)
    plt.close()
    return


def plot_convergence(dists, output_dir='/'):
    '''
    Plot convergence rate for all participants across all steps
    '''
    plt.clf()
    fig = plt.figure()

    # caclulate the mean and std at the distance at the last step
    last_dists = [dist[-1] for dist in dists]
    last_dist_mean, last_dist_std = np.mean(last_dists), np.std(last_dists)
    fig.suptitle(f'Distance (mean $\pm$ std): {round(last_dist_mean, 3)} $\pm$ {round(last_dist_std, 3)}') 

    # plot 95% confidence interval
    means = np.mean(dists, axis=0)
    stds = np.std(dists, axis=0)
    x_ticks = [x + 1 for x in range(len(dists[-1]))]
    plt.fill_between(x_ticks, means - 2*stds, means + 2*stds, alpha = 0.2, color='black', label='95% CI') 
    
    # plot individual convergence plots
    for i in range(len(dists)):
        num_its = len(dists[i])
        plt.plot( [x + 1 for x in range(num_its)] , dists[i], alpha=0.3, label=str(i+1))
    
    plt.ylabel('Dist')
    plt.xlabel('Iterations')
    plt.ylim([-0.5, 1.25])
    # plot mean
    plt.plot(x_ticks, np.mean(dists, axis=0), linewidth=2, color='blue', label='Mean')
    plt.tight_layout(rect=[0, 0, 0.8, 1.0])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(output_dir + 'convergence.png', dpi=300)
    plt.close()
    

def plot_verification(random_par_dist, true_par_dist, sel_par_dist, output_dir='/'):
    plt.bar(['Random model', 'True model', 'Estimated model'], \
        [np.mean(random_par_dist), np.mean(true_par_dist), np.mean(sel_par_dist) ])
    plt.tight_layout()
    plt.savefig(output_dir + 'behavioral_fit.png', dpi=300)
    plt.close()
    pass


def plot_behavioral_fit(rand_model_vers, true_model_vers, est_model_vers, output_dir='/'):
    data = np.array([rand_model_vers, true_model_vers, est_model_vers])
    print('Verification data: ', data)

    fig, ax=plt.subplots()
    for i in range(data.shape[1]):
        bottom=np.sum(data[:,0:i], axis=1)  
        ax.bar(['Random model', 'True model', 'Estimated model'], data[:,i], bottom=bottom, label="label {}".format(i))

    plt.tight_layout()
    plt.savefig(output_dir + 'behavioral_fit.png', dpi=300)
    plt.close()
    pass


def plot_behavioral_convergence(fitness_dists, output_dir='/'):
    print(np.array(fitness_dists).shape)
    participant_mean = np.mean(fitness_dists, 0)
    participant_std = np.std(fitness_dists, 0)

    transposed_means = np.transpose(participant_mean)
    transposed_stds = np.transpose(participant_std)
    
    x_ticks = range(transposed_means.shape[1])

    model_labels = ['Random model', 'True model', 'Estimated model']
    colors = ['orange', 'blue', 'green']
    for label, i in zip(model_labels, range(len(model_labels))):
        plt.plot(x_ticks, transposed_means[i,:], label=label)
        plt.fill_between(x_ticks, transposed_means[i,:]-2*transposed_stds[i,:], \
            transposed_means[i,:]+2*transposed_stds[i,:], color='grey', alpha=0.2)
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_dir + 'behavioral_convergence.png', dpi=300)
    plt.close()
    pass


def plot_designs(designs, design_bounds, output_dir='/'):
    plt.clf()
    fig = plt.figure()

    for i in range(len(designs)):
        num_its = len(designs[i])
        plt.scatter( [x+1 for x in range(num_its)], designs[i], alpha=0.3, label=str(i+1))

    plt.ylim(design_bounds)
    plt.ylabel('Designs')
    plt.xlabel('Iterations')
    plt.tight_layout(rect=[0, 0, 0.8, 1.0])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(output_dir + 'designs.png', dpi=300)
    plt.close()


def plot_model_results(model_posteriors, true_models, output_dir='/'):
    df = pd.DataFrame(model_posteriors)
    vals = np.around(df.values,2)
    colours = plt.cm.hot(vals)
    true_models = [ f'{number+1}. {name} ' for name, number in zip(true_models, range(len(true_models)))]

    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
    the_table=plt.table(cellText=vals, rowLabels=true_models[:len(df.index)], 
            colLabels=df.columns, colWidths = [0.03]*vals.shape[1], loc='center', 
            cellColours=colours)

    plt.tight_layout()
    plt.savefig(output_dir + 'model_results.png', dpi=300)
    plt.close()


def plot_model_convergence(model_posteriors, participant):
    output_dir = participant.hyper['output_dir']
    os.makedirs( output_dir , exist_ok=True ) 

    model_names = list(model_posteriors[-1].keys())
    for model_name in model_names:
        model_prob = np.array([models_at_step[model_name] for models_at_step in model_posteriors])
        print(model_prob)
        plt.plot(range(len(model_posteriors)), model_prob, label=model_name)
    plt.title('True model: ' + participant.name)
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_dir + 'model_posterior_convergence.png', dpi=300)
    plt.close()


def plot_GP_mean(sim_bounds, gp, X, Y, output_dir=None, it=None, true=None, est=None, expt=None, thr=None):
    ax = plt.clf()
    ax = plt.gca()
    
    test_x = np.linspace(*sim_bounds, num = 100)
    mean, var = gp.predict(test_x)
    lower = np.array(mean - 3*var).flatten()
    upper = np.array(mean + 3*var).flatten()
    #print(mean,lower, upper)
    ax.plot(X, Y, 'k*')

    ax.axvline(est, ls='dashed', color="black", label='Estimate')
    ax.axvline(true, ls='-', color="red", label='Ground Truth')
    ax.axhline(thr, ls='-.', color='blue', label='Threshold')

    ax.plot(test_x, mean, 'b')
    ax.fill_between(test_x, lower, upper, color='grey', alpha=0.5)

    plt.xlabel(r"par")
    plt.ylabel(r"Log Discrepancy")
    plt.legend()
    plt.title('Design: ' + str(expt))
    output_dir = output_dir + 'theta_log/'
    os.makedirs( output_dir , exist_ok=True ) 
    plt.savefig(output_dir + 'GP-' + str(it) + '.png', dpi=300, bbox_inches="tight")
    plt.close()


def show_design_GP(model, train_x, train_obj, bounds):
    ax = plt.clf()
    ax = plt.gca()
    # test model on 101 regular spaced points on the interval [0, 1]
    test_X = torch.linspace(bounds[0], bounds[1], 100)
    # no need for gradients
    with torch.no_grad():
        # compute posterior
        posterior = model.posterior(test_X)
        # Get upper and lower confidence bounds (2 standard deviations from the mean)
        lower, upper = posterior.mvn.confidence_region()
        # Plot training points as black stars
        ax.plot(train_x, train_obj, 'k*')
        # Plot posterior means as blue line
        ax.plot(test_X.cpu().numpy(), posterior.mean.cpu().numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_X.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)
        # TODO: we need to maintain a posterior over designs, 
    plt.show()


def print_logs(all_dict, output_dir='/'):
    sns.set_theme()
    sns.color_palette()
    colors = ['orange', 'green', 'purple', 'brown', 'pink', 'olive', 'cyan']
    legends = {'sp': 'BOSMOS', 'true_lik': 'True lik.',  'true_lik_rand': 'EIRD', \
        'sp_rand': 'SP (rand d)', 'ado': 'ADO', 'minebed': 'MINEBED', 'sp_bic': 'BOSMOS (BIC)', \
        'minebed_bic': 'MINEBED (BIC)'}
        
    table_ticks = [0, 1, 3, 19, 99]

    # ===================================
    # statistics
    # ===================================
    print('\n\n 1. behavior fitness: (dist, trials) ')
    fig = plt.figure()
    for i, filename in zip(range(len(all_dict)), all_dict):
        print('')
        print(filename)
        if str(filename) == 'sp':
            rand_model_fitness = all_dict[filename]['rand_model_fitness'].flatten()
            true_model_fitness = all_dict[filename]['true_model_fitness'].flatten()
            plt.axhline(y=np.mean(rand_model_fitness), color='r', linestyle='-.', label='Random model')
            plt.axhline(y=np.mean(true_model_fitness), color='b', linestyle='-.', label='True model')
            
        rand_model_fitness = all_dict[filename]['rand_model_fitness'].flatten()
        est_fitness_traj = all_dict[filename]['fitness_trajectories'][:, :, 2].transpose()  
        x_ticks = range(1, len(est_fitness_traj) + 1)

        mean, std = np.mean(est_fitness_traj, axis=1), np.std(est_fitness_traj, axis=1)
        # print(np.shape(mean))
        plt.plot(x_ticks, np.mean(est_fitness_traj, axis=1), color=colors[i], label=legends[filename])
        plt.fill_between(x_ticks, mean - std, mean + std, color=colors[i], alpha=0.1)
        # print('Behavioral fitness (mean +- std):')
        # print( [r"{mean:.2f} $\pm$ {std:.2f}".format(mean=m, std=s) for item_id, m, s in zip(range(len(mean)), mean, std) if item_id in table_ticks ])
        
        dist_traj = all_dict[filename]['dist_trajectories'].transpose()[:, 1:]
        all_model_choices = all_dict[filename]['all_model_choices'].transpose()[:, 1:]
        true_models = all_dict[filename]['true_models']
        true_models = true_models[:np.shape(all_model_choices)[1]]
        mask = []
        temp_model_choices = np.array(all_model_choices, copy=True).transpose()
        for model_choices, true_model_name in zip(temp_model_choices, true_models):
            mask.append( [ true_model_name == model_name for model_name in model_choices ] )
        mask = np.array(mask).transpose()
        
        mean, std = np.mean(dist_traj, axis=1), np.std(dist_traj, axis=1)
        print('Complete parameter estimate error (mean +- std):')
        print( [r"{mean:.2f} $\pm$ {std:.2f}".format(mean=m, std=s) for item_id, m, s in zip(range(len(mean)), mean, std) if item_id in table_ticks ])
        
        dist_traj[~mask] = np.nan
        print(str(dist_traj))
        
        mean, std = np.nanmean(dist_traj, axis=1), np.nanstd(dist_traj, axis=1)
        print('Parameter estimate error (mean +- std):')
        print( [r"{mean:.2f} $\pm$ {std:.2f}".format(mean=m, std=s) for item_id, m, s in zip(range(len(mean)), mean, std) if item_id in table_ticks ])
        
        print('Model accuracy:')
        model_keys = all_dict[filename]['model_posterior_keys'].flatten()
        for key in model_keys:
            model_mask = np.array([true_model_name == key for true_model_name in true_models]) # * np.shape(mask)[0]  )
            model_accuracy = [ np.sum(row.astype(int) * model_mask.astype(int)) for row in mask]  # np.dot(mask.astype(int),) #  np.sum(model_mask * mask, axis=1)
            # print(np.shape(model_mask), len(model_accuracy), np.sum(model_mask), np.sum(mask), model_mask, model_accuracy)
            print(key)
            print( [r'{m:.2f}'.format(m=float(m)/ np.sum(model_mask)) for item_id, m in zip(range(len(model_accuracy)), model_accuracy) if item_id in table_ticks ])
        
        # print(np.shape(dist_traj), dist_traj[-1])
        print(np.shape(all_model_choices), all_model_choices[-1])
        print(np.shape(true_models), true_models)
        # print(np.shape(mask), mask[-1])
        
        times = all_dict[filename]['time'].flatten()
        m, s = np.mean(times), np.std(times)
        print( r'Time: {mean:.2f} $\pm$ {std:.2f}'.format(mean=m, std=s)  )

    plt.xlim([1, len(est_fitness_traj)])
    plt.xlabel('Trials (t)')
    plt.ylabel('RMSE')
    plt.tight_layout()
    
    handles,labels = plt.gca().get_legend_handles_labels()
    sorted_legends = [x for x in sorted(labels)] # sort the labels based on the average which is on a list
    sorted_handles = [x for _, x in sorted(zip(labels, handles))] # sort the handles based on the average which is on a list
    plt.legend(sorted_handles,sorted_legends) # display the handles and the labels on the side
    
    # plt.savefig(output_dir + 'plot-1.png', dpi=300)
    plt.close()

