library(fase)
library(readr)
library(abind)
library(pROC)


set.seed(1)

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
args = commandArgs(trailingOnly = TRUE)
seed = args[1]
n_nodes = as.numeric(args[2])
n_time_points = as.numeric(args[3])
density = as.numeric(args[4])

dir_name = paste0('fase_data')

time_points = readr::read_table(paste0(dir_name, '/time_points.npy'), col_names = FALSE)$X1
A = NULL
X = NULL
for (t in 1:length(time_points)) {
    A = abind(A, read.table(paste0(dir_name, '/Y_', t, '.npy')), along = 3) 
    X = abind(X, read.table(paste0(dir_name, '/X_', t, '.npy')), along = 3)
}


k_steps = 5
n_insample_steps = length(time_points) - k_steps
idx = 1
qs = seq(4, min(10, length(time_points)-1), by = 2)
model_select = matrix(0, nrow = length(qs),  ncol = 3)
fits = list()
start.time <- Sys.time()
for (q in qs) {
    fit <- fase(A[,,1:n_insample_steps], d=2, self_loops=FALSE,
            spline_design=list(type='bs', q=q, x_vec=time_points[1:n_insample_steps]),
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

Z_align = proc_align_slicewise3(fit$Z, X[,,1:n_insample_steps])

Z_mse = mean((Z_align - X[,,1:n_insample_steps]) ^ 2)

# forecast
Z_forecast = Z_align[,,n_insample_steps]
proba_forecast = pmin(pmax(tcrossprod(Z_forecast), 0), 1)
subdiag = lower.tri(proba_forecast, diag = FALSE)
proba_forecast = proba_forecast[subdiag]
forecast_mse = numeric(k_steps+1)
for (k in 0:k_steps) {
    X_true = X[,,n_insample_steps + k]
    true_proba = pmin(pmax(tcrossprod(X_true), 0), 1)[subdiag]
    forecast_mse[k+1] = mean((proba_forecast - true_proba) ^ 2)
}

data = data.frame(
    fase_mse = Z_mse,
    fase_kstep_0 = forecast_mse[1],
    fase_kstep_1 = forecast_mse[2],
    fase_kstep_2 = forecast_mse[3],
    fase_kstep_3 = forecast_mse[4],
    fase_kstep_4 = forecast_mse[5],
    fase_kstep_5 = forecast_mse[6]
)
print(data)

#out_file = paste0('result.csv')
#dir_name = paste0('output')
#dir.create(dir_name)
#write_csv(data, paste0(dire_name, '/', out_file))
