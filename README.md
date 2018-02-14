# ug_thesis

### Compilation
`./compile.sh`

### Inputs
* COLMAP-generated 3D sparse reconstruction outputs `points3D.txt` and `database.db`

### Usage
* To load from COLMAP outputs `points3D.txt` and `database.db`, with input image config file `config.ini`:

   `./main <path_to_points3D.txt> <path_to_database.db> <path_to_config.ini>`

* To load from COLMAP outputs `points3D.txt` and `database.db`, and save to `serialised_data.txt`, with input image config file `config.ini`:

   `./main <path_to_points3D.txt> <path_to_database.db> <path_to_serialised_data.txt> <path_to_config.ini>`

* To load from previously saved `serialised_data.txt`, with input image config file `config.ini`:

   `./main <path_to_serialised_data.txt> <path_to_config.ini>`
   
### Dependencies
* OpenCV (including non-free)
* Eigen
* Boost (Serialisation)
* Ceres Solver
* COLMAP
