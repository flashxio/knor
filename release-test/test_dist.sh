OUTDIR_IM=outdir-IM
OUTDIR_DM=outdir-DM
OUTDIR_FDM=outdir-FDM

cd .. &&\
    exec/knori test-data/matrix_r50_c5_rrw.bin 50 5 8 \
     -t none -C test-data/init_clusters_k8_c5.bin -T 2 -o $OUTDIR_IM && \
 mpirun.mpich -n 2 exec/knord test-data/matrix_r50_c5_rrw.bin 50 5 8 \
    -t none -C test-data/init_clusters_k8_c5.bin -T 2 -o $OUTDIR_DM &&
    if [ "$(diff $OUTDIR_IM/* $OUTDIR_DM/*)" = "" ];
then
    echo "knord PRUNED test success!"
else
    echo "knord PRUNED release test failure!"
    exit 1
fi

mpirun.mpich -n 2 exec/knord test-data/matrix_r50_c5_rrw.bin 50 5 8 \
    -t none -C test-data/init_clusters_k8_c5.bin -T 2 -P -o $OUTDIR_FDM &&
    if [ "$(diff $OUTDIR_IM/* $OUTDIR_FDM/*)" = "" ];
then
    echo "knord FULL test success!"
else
    echo "knord FULL release test failure!"
    exit 1
fi

# Cleanup
echo "Cleaning up ..."
rm -rf $OUTDIR_IM $OUTDIR_DM $OUTDIR_FDM
