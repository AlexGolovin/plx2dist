cff-version: 1.2.0
message: "If you use this software, please cite it using the metadata below."
title: "plx2dist: Bayesian parallax-to-distance pipeline"
version: "0.1.0"
date-released: "2025-01-01"      # update to actual release date
license: MIT
repository-code: "https://github.com/YOUR_USERNAME/plx2dist"
doi: "10.5281/zenodo.XXXXXXX"   # fill in after Zenodo registration

authors:
  - family-names: Golovin
    given-names: Alex
    orcid: "https://orcid.org/0000-0000-0000-0000"   # replace with your ORCID
    affiliation: "YOUR INSTITUTION"

abstract: >
  plx2dist computes Bayesian geometric distances from trigonometric parallax
  measurements. It uses an adaptive two-stage log-r grid to evaluate
  p(r | parallax, sigma) under EDSD or uniform volume density priors,
  returning posterior quantiles, mode, and membership probabilities.
  Developed for CNS6 (Catalogue of Nearby Stars, 6th edition).

keywords:
  - astronomy
  - astrometry
  - parallax
  - stellar distances
  - Gaia
  - Bayesian inference
  - solar neighbourhood

references:
  - type: article
    title: "The Catalogue of Nearby Stars. Fifth Edition (CNS5)"
    authors:
      - family-names: Golovin
        given-names: Alex
    journal: "Astronomy & Astrophysics"
    volume: 670
    pages: A19
    year: 2023
    doi: "10.1051/0004-6361/202244250"