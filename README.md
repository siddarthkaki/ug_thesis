# ug_thesis

### Compilation
`./compile.sh`

### Inputs
* COLMAP-generated 3D sparse reconstruction outputs `points3D.txt` and `database.db`

### Usage
* To load from COLMAP outputs `points3D.txt` and `database.db`:

   `./main <path_to_points3D.txt> <path_to_database.db>`

* To load from COLMAP outputs `points3D.txt` and `database.db`, and save to `serialised_data.txt`:

   `./main <path_to_points3D.txt> <path_to_database.db> <path_to_serialised_data.txt>`

* To load from previously saved `serialised_data.txt`:

   `./main <path_to_serialised_data.txt>`
   
### Dependencies
* OpenCV (including non-free)
* Eigen
* Boost (Serialisation)
* Ceres Solver
* COLMAP
