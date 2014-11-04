Volumetric path tracer
========

This is a global illuminated volume renderer featuring Woodcock tracking & importance sampled path tracing.

The Woodcock tracking technique distinguishes from the raymarching technique in that it does not approach the integral of the volume rendering equation with a biased numerical integration (using composite Simpson quadrature, or by trapezoidal rule, etc), but rather applies the Monte Carlo technique for unbiased integration. This allows the global illumination solution of the volume to be unbiased and strong, especially when there's heavy spectral dependence in the medium's scattering properties.

Albeit imaginary, the physical explanation of this approach is to introduce an ideal auxiliary component that has no absorption but perfect forward scattering, thus not altering the direction or energy of the scattered light rays. In such a mixed medium, scattering events are decided probabilistically by the volume fraction of the corresponding component. And an effective scattering event only occurs when the normal component is hit.

In a naive implementation, the overall density of the mixed medium is uniform, and the Monte Carlo technique of sampling transmittance in a homogeneous medium is applied. Advanced techniques try to combine the raymarching method for stochastically sampling transmittance in heterogeneous media and Woodcock technique for scattering events.

Woodcock tracking is sometimes referred to as rejection sampling. But this is due to the misunderstandings that the rejected samples in a Woodcock sampling approach are invalid. A usual scenario of applying the rejection sampling technique would be to uniformly sample a disk by first sampling its bounding box, and in such a case both accepted and rejected samples are independent. But in Woodcock tracking, the discarded samples are actually buliding blocks for the effective samples, and are not rejected conceptually.
