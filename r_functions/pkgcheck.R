pkgcheck <- function( pkg ) {

  # Check wich packages are not intalled and install them
  if ( sum(!pkg %in% installed.packages()[, 'Package'])) {
    install.packages(pkg[ which(!pkg %in% installed.packages()[, 'Package'])],
                     dependencies = T)
  }
  
  # Require all packages
  sapply(pkg, require, character.only =  T)
  
}