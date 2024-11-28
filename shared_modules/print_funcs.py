#
import torch
from types import SimpleNamespace

def check_namespaces(namespace_one, namespace_two, print_equal=False):

    # 1. Extract keys
    namespace_one_dict = namespace_one.__dict__
    namespace_two_dict = namespace_two.__dict__

    # 2. Find matches
    list_of_matches = []

    for key in namespace_one_dict:
        if key in namespace_two_dict:
            list_of_matches.append(key)

    # 3. Check if values are the same
    not_equal = []
    for match in list_of_matches:
        if namespace_one_dict[match] == namespace_two_dict[match]:
            if print_equal:
                print(f'{match} is the same in both namespaces')
            
        else:
            print(f'{match} is NOT the same in both namespaces: {namespace_one_dict[match]} not equal to {namespace_two_dict[match]}')
            not_equal.append(match)

    # 4. Print, inform user
    if not_equal:
        print(f'\nThe following differences in attributes have been detected:')
        print(f'{"-"*8} Namespace one {"-"*33} Namespace two {"-"*33}')
        for unequal in not_equal:
            print(f'{unequal}\t {namespace_one_dict[unequal]} \t \t \t {namespace_two_dict[unequal]}')
    
    else:
        print('All matched attributes have the same value')

def module_information(module):
    attributes = [attr for attr in dir(module) if not attr.startswith(('__', '_'))]
    attributes_dict = {}

    namespace_attributes = ['self', 'par', 'sol', 'sim', 'allocate', 'define_parameters', 'allocate_memory', 'training', 't_nn']
    module_attributes = ['datetime', 'np', 'plt', 'allocate ']
    other_attributes = ['as_dict', 'check_types', 'copy', 'cpp', 'cpp_filename', 'cpp_options', 'cpp_structsmap', 'from_dict', 'infer_types', 'internal_attrs', 'link_to_cpp', 'load', 'name', 'namespaces', 'not_floats', 'other_attrs', 'save', 'savefolder', 'settings', 'setup', 'update_jit']
    not_wanted_attributes = namespace_attributes + module_attributes + other_attributes

    for attr in attributes:
        bool = attr in not_wanted_attributes

        if not bool:
            try:
                func_unique_vars = list(module.__getattribute__(attr).__code__.co_varnames[:module.__getattribute__(attr).__code__.co_argcount])

            except:
                pass

            func_unique_vars_wanted = []

            for var in func_unique_vars:

                if not var in not_wanted_attributes:
                    func_unique_vars_wanted.append(var)
            
            attributes_dict[attr] = func_unique_vars_wanted
    
    module_name = str(module)

    if len(module_name) > 30:
        module_name = module_name[:15]

    print(f'{"-"*120}')
    print(f'\nModule: \t\t{module_name} has the following attributes:')
    print(f'\nAttribute\t\t\tArguments')
    print(f'{"-"*120}')
    for attr in attributes_dict:
        if attributes_dict[attr]:
            attr_args = "\t".join(f'{arg:<10}' for arg in attributes_dict[attr]) + '\n'
            print(f"{attr:<32}{attr_args:>10}")
        else:
            print(f"{attr:<50}{'function takes no arguments':>20}\n") 