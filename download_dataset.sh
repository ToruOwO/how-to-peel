### apple

# download data files
curl -L -O "https://www.dropbox.com/scl/fi/90mvjwv77bvihcdxsh0d3/apple.tar.part-aa?rlkey=kyb3ep433jdqseqqfc4uao0z7&dl=1"
curl -L -O "https://www.dropbox.com/scl/fi/ggsadxg0gono47yclrrlm/apple.tar.part-ab?rlkey=lesd2ryy1ou7bw9c22bs89xcu&dl=1"
curl -L -O "https://www.dropbox.com/scl/fi/d462stkoxe1nv6jaxggih/apple.tar.part-ac?rlkey=jughfwfxyg8y0h6hamgyv3tm4&dl=1"

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