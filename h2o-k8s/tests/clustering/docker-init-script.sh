#! /bin/bash -x

pwd
cd $PWD/h2o-k8s/tests/clustering/
ls
k3d --version
k3d delete
k3d create -v $PWD/build/h2o.jar:$PWD/build/h2o.jar --publish 8080:80 --api-port localhost:6444 --server-arg --tls-san="127.0.0.1" --wait 120 
export KUBECONFIG="$(k3d get-kubeconfig --name='k3s-default')"
kubectl cluster-info
sleep 15 # Making sure the default namespace is initialized. The --wait flag does not guarantee this.
kubectl get namespaces
envsubst < h2o-service.yaml >> h2o-service-subst.yaml
kubectl apply -f h2o-service-subst.yaml
kubectl wait --for=condition=available --timeout=600s deployment.apps/h2o-deployment -n default
rm h2o-service-subst.yaml
kubectl describe pods
timeout 120s bash h2o-cluster-check.sh
export EXIT_STATUS=$?
kubectl get pods
kubectl get nodes
k3d delete
exit $EXIT_STATUS
