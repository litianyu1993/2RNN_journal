import argparse
'''Parser set up'''
parser = argparse.ArgumentParser()

'''General experiment specification'''
parser.add_argument('-exp', '--experiment_name', help='name of experiments, Addition, RandomRNN or Wind')
parser.add_argument('-lne', '--list_number_examples', nargs='+', help='list of examples numbers', type=int)
parser.add_argument('-nr', '--number_runs', help='number of runs', type=int)
parser.add_argument('-le', '--length', help='minimum training length', type=int)
parser.add_argument('-tle', '--testing_length', help='testing length', type=int)
parser.add_argument('-xp', '--xp_path', help='experiment folder path')
parser.add_argument('-lm', '--method_list', nargs='+',
                    help="List of methods to use, can be IHT, TIHT, NuclearNorm, OLS, ALS, LSTM, SGD+TIHT")

'''If using TIHT/IHT specify the following'''
parser.add_argument('-eps', '--HT_epsilon', help='epsilon for TIHT and IHT', type=float)
parser.add_argument('-lr', '--HT_learning_rate', help='learning rate for TIHT and IHT', type=float)
parser.add_argument('-mi', '--HT_max_iter', help='number of max iterations for TIHT and IHT', type=int)

'''If using NuclearNorm method, specify the following'''
parser.add_argument('-a', '--alpha', help='hyperparameter for nuclear norm method', type=float)

'''If using ALS, specify the following'''
parser.add_argument('-aepo', '--ALS_epoches', help='Number of epochs when using ALS', type=int)

'''If running Random2RNN exp, and launching a new experiment, specify the following'''
parser.add_argument('-var', '--noise', help='variance of the gaussian noise', type=float)
parser.add_argument('-ns', '--states_number', help='number of states for the model', type=int)
parser.add_argument('-ld', '--load_data', default=False, help='load the previously created data', action='store_true')
parser.add_argument('-tfn', '--target_file_name', help='target file name')
parser.add_argument('-tns', '--target_number_states', help='number of states for the target 2-rnn', type=int)
parser.add_argument('-tid', '--target_input_dimension', help='input dimension for the target 2-rnn', type=int)
parser.add_argument('-tod', '--target_output_dimension', help='output dimension for the target 2-rnn', type=int)
parser.add_argument('-lt', '--load_target', default=False, help='load the previously created target 2rnn',
                    action='store_true')

'''If running TIHT+SGD, specify the following'''
parser.add_argument('-lr2', help='learning rate for sgd 2rnn', type=float)
parser.add_argument('-epo2', help='number of epochs for sgd 2rnn', type=int)
parser.add_argument('-b2', '--batch_size', help='batch size for sgd 2rnn', type=int)
parser.add_argument('-t', '--tolerance', help='tolerance for sgd 2rnn', type=int)
args = parser.parse_args()

'''Arguments set up'''
if args.experiment_name != None:
    exp = args.experiment_name
else:
    raise Exception('Did not initialize which experiment to run, try set up after -exp argument')
if args.noise != None:
    noise_level = args.noise
else:
    raise Exception('Did not initialize noise_level, try set up after -var argument')
if args.states_number != None:
    num_states = args.states_number
else:
    raise Exception('Did not initialize state numbers, try set up after -ns argument')
if args.alpha != None:
    alpha = args.alpha
else:
    raise Exception('Did not initialize alpha, try set up after -a argument')

if args.load_data:
    load_data = True
else:
    load_data = False

if args.load_target:
    load_target = True
else:
    load_target = False

if args.list_number_examples:
    L_num_examples = args.list_number_examples
if args.number_runs:
    N_runs = args.number_runs
if args.length:
    length = args.length
if args.ALS_epoches:
    ALS_epochs = args.ALS_epoches
if args.testing_length:
    test_length = args.testing_length
if args.method_list:
    methods = args.method_list
if args.HT_epsilon:
    TIHT_epsilon = args.HT_epsilon
if args.HT_learning_rate:
    TIHT_learning_rate = args.HT_learning_rate
if args.HT_max_iter:
    TIHT_max_iters = args.HT_max_iter
if args.xp_path:
    xp_path = args.xp_path

if args.lr2:
    lr2 = args.lr2
if args.epo2:
    epo2 = args.epo2
if args.batch_size:
    b2 = args.batch_size
if args.tolerance:
    tol = args.tolerance

if args.target_file_name:
    target_file_name = args.target_file_name
if args.target_number_states:
    target_num_states = args.target_number_states
if args.target_input_dimension:
    target_input_dim = args.target_input_dimension
if args.target_output_dimension:
    target_output_dim = args.target_output_dimension

method_name(num_states, noise_level, load_data, load_target)