
__kernel void vadd( 
    __global float *a,
    __global float *b,
    __global float *c,
    const unsigned int count)
{
    int index = get_global_id(0);
    if (index < count)
        c[index] = a[index] + b[index];
}