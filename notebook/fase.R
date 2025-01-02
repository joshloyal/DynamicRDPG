library(fase)
library(readr)
library(abind)
library(pROC)


set.seed(1)

logit = function(x) {
    log(x/(1-x))
}

## Helper functions taken from the FASE package

# hollowization for square matrices
hollowize <- function(M){
    M - diag(diag(M))
}

# hollowization for 3D arrays
hollowize3 <- function(A){
    array(apply(A,3,hollowize),dim(A))
}

Z_to_Theta <- function(Z,self_loops = FALSE){
    Theta <- array(apply(Z,3,tcrossprod),
                   c(dim(Z)[1],dim(Z)[1],dim(Z)[3]))
    if(self_loops){
        return(Theta)
    }
    else{
        return(hollowize3(Theta))
    }
}


## Run comparison

dir_name = paste0('fase_data')

time_points = readr::read_table(paste0(dir_name, '/time_points.npy'), col_names = FALSE)$X1
A = NULL
X = NULL
for (t in 1:length(time_points)) {
    A = abind(A, read.table(paste0(dir_name, '/Y_', t, '.npy')), along = 3) 
    X = abind(X, read.table(paste0(dir_name, '/X_', t, '.npy')), along = 3)
}


idx = 1
qs = seq(4, 24, by = 2)
model_select = matrix(0, nrow = length(qs),  ncol = 3)
fits = list()
start.time <- Sys.time()
for (q in qs) {
    fit <- fase(A,d=2,self_loops=FALSE,
            spline_design=list(type='bs',q=q,x_vec=time_points),
            output_options=list(return_coords=TRUE))
    model_select[idx,] = c(idx, q, fit$ngcv)
    print(paste0('q = ', q, ' ngcv = ', fit$ngcv))
    fits[[idx]] = fit
    idx = idx + 1
}
end.time <- Sys.time()
time_fase = end.time - start.time

best_idx = which.min(model_select[,3])
fit = fits[[best_idx]]

Z_align = proc_align_slicewise3(fit$Z, X)

mean((Z_align - X) ^ 2)
