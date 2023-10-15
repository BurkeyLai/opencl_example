__kernel void mmul(
    int N,
    __global float *a,
    __global float *b,
    __global float *c)
{
    int tx = get_global_id(0); // column
    int ty = get_global_id(1); // row
    float value = 0;

    for (unsigned int i = 0; i < N; ++i) {
        value += a[ty * N + i] * b[i * N + tx];
    }
    c[ty * N + tx] = value;

    /*
    int index = get_global_id(0);
    if (index < N)
        c[index] = a[index] + b[index];
    */
}
