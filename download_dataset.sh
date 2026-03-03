### apple

# download data files
curl -L -O "https://www.dropbox.com/scl/fi/90mvjwv77bvihcdxsh0d3/apple.tar.part-aa?rlkey=kyb3ep433jdqseqqfc4uao0z7&dl=1"
curl -L -O "https://www.dropbox.com/scl/fi/ggsadxg0gono47yclrrlm/apple.tar.part-ab?rlkey=lesd2ryy1ou7bw9c22bs89xcu&dl=1"
curl -L -O "https://www.dropbox.com/scl/fi/d462stkoxe1nv6jaxggih/apple.tar.part-ac?rlkey=ahzf1k1eglzw6ga6nloio85es&dl=1"

# verify integrity
zstd -t apple.tar.zst

# concatenate and extract
cat apple.tar.part-* | tar -I zstd -xf -


### cucumber

# download data files
curl -L -O "https://www.dropbox.com/scl/fi/enr3g9uzei98vedzdir1r/cucumber.tar.part-aa?rlkey=7evqwxvd4ae3gfo7tw5k9er0i&dl=1"
curl -L -O "https://www.dropbox.com/scl/fi/ixb88nlxl6xxrska4ih8m/cucumber.tar.part-ab?rlkey=qbnatoy17mmoll76sx3ky5qnz&dl=1"

# verify integrity
zstd -t cucumber.tar.zst

# concatenate and extract
cat cucumber.tar.part-* | tar -I zstd -xf -


### potato (COMING SOON)

# download data files
curl -L -O "https://www.dropbox.com/scl/fi/zxij0wwqax3za8qsw41e8/potato.tar.part-aa?rlkey=tcbwwaamm690ok1x12qgil787&dl=1"
curl -L -O "https://www.dropbox.com/scl/fi/cddufj8q33rw0waejr2qu/potato.tar.part-ab?rlkey=24d44uqgog9ptxibfnm1v4v81&dl=1"
curl -L -O "https://www.dropbox.com/scl/fi/kq7mr03fwt0ejufokl915/potato.tar.part-ac?rlkey=zpwlgygqy860box1vn766efpj&dl=1"
curl -L -O "https://www.dropbox.com/scl/fi/sq7d6c2w7o5byz6hl1f46/potato.tar.part-ad?rlkey=14ykbb54v1k9sxs5cjduxryyc&dl=1"
curl -L -O "https://www.dropbox.com/scl/fi/jwy7nw88cutgk6g8onf7e/potato.tar.part-ae?rlkey=d424lqdwq0ta1fq3ys9pmv0ok&dl=1"
curl -L -O "https://www.dropbox.com/scl/fi/i8c5ke4umbammohsteoyy/potato.tar.part-af?rlkey=04os4t5srfptrsb4a2p9r66gq&dl=1"

# verify integrity
zstd -t potato.tar.zst

# concatenate and extract
cat potato.tar.part-* | tar -I zstd -xf -