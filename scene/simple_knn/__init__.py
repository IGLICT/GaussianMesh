import os 
import jittor as jt

header_path = os.path.join(os.path.dirname(__file__), 'cuda_headers')
if os.path.dirname(__file__) == '':
    lib_path = "."
else:
    lib_path = os.path.join(os.path.dirname(__file__))
proj_options = {f'FLAGS: -I"{header_path}" -l"simpleknn" -L"{lib_path}/"':1}

cuda_header = """
#include "simple_knn.h"
"""
jt.flags.use_cuda = 1
def distCUDA2(points:jt.Var) -> jt.Var:
    P = points.size(0)
    means = jt.zeros((P,),'float32')
    with jt.flag_scope(compile_options=proj_options):
        means, = jt.code(outputs = [means],inputs = [points], cuda_header=cuda_header,
        cuda_src=f"""
            @alias(points, in0)
            @alias(means, out0)

            SimpleKNN::knn({P}, (float3*)points_p, means_p);
        
        """)
    # means.sync()
    return means
