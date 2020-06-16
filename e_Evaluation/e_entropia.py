from subprocess import Popen, PIPE, STDOUT


def compute_entropia(variant, xes_file, pnml_file):
    args = ['java', '-jar', 'e_Evaluation/jbpt-pm-entropia-1.6.jar', f'-{variant}', '-srel=3',
            '-sret=3',
            f'-rel={xes_file}', f'-ret={pnml_file}', '--silent']
    p = Popen(args, stdout=PIPE, stderr=STDOUT)

    res = ""
    for line in p.stdout:
        res = res + str(line)

    return res
