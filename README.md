Volumetric path tracer
========

This is a global illuminated volume renderer featuring Woodcock tracking & importance sampled path tracing.

The Woodcock tracking technique distinguishes from the raymarching technique in that it does not approach the integral of the volume rendering equation with a biased Riemann sum, but rather applies the Monte Carlo technique for unbiased integration. This allows the global illumination solution of the volume to be unbiased and strong, especially when there's heavy spectral dependence in the medium's scattering properties.
