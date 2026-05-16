# SIAM CSE 2025
Fast direct solvers Minitutorial materials

## Overview

This repo contains the talks and presentations from the SIAM CSE minitutorial on Fast Direct Solvers.  The presentations are available via pdfs titled:
 'FDS_tutorial_part1.pdf' and 'FDS_tutorial_part2.pdf'.  The first minitutorial focuses on solvers for sparse linear systems while the second tutorial focuses on solvers for dense linear systems.  Some particular attention is made on compression techniques.  
 
 For each minitutorial session, there is a folder with corresponding demo codes that help illustrate properties of the matrices involved and the techniques that are used to build the fast solvers.  Note that none of the codes provides are intended to be production level.  They are provided as easy to interact with illustrations of the ideas behind the solvers.
 
 The demonstration codes in the 'Part 2 demos' folder utilize the newly available randomized linear algebra package called 'librla' that is available via the information provided below.  The provided codes already utilize the linear algebra package.   Please ensure to reference 'librla' appropriately if you use these codes.  To utilize the provided 'Python' codes, please use the 'setup-venv.sh' file which sets up a virtual Python environment and installs the necessary packages.
 
## References
A brief list of related materials used to create these tutorials and demo codes is provided here:


- Gillman, Young, Martinsson. "A direct solver with O(N) complexity for integral equations on one-dimensional domains." Frontiers of Mathematics in China 7 (2), 217-247.
- Martinsson. "Fast direct solvers for elliptic PDEs" Society for Industrial and Applied Mathematics.
- ['librla:Randomized Linear Algebra Library' ](https://github.com/agillman20/librla)

## License

See License file in the repository.

## Contact

For questions, issues, or contributions, please contact the repository maintainers.
