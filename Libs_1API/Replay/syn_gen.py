out_idx_tensor="torch::tensor({"
out_value_tensor="torch::tensor({"
batchsize=64
for i in range(batchsize):
    if (i!=batchsize-1):
        out_idx_tensor+="i*"+str(batchsize)+"+"+str(i)+","
        out_value_tensor+="0.1*(i*"+str(batchsize)+"+"+str(i)+"),"
    else:
        out_idx_tensor+="i*"+str(batchsize)+"+"+str(i)+"})"
        out_value_tensor+="0.1*(i*"+str(batchsize)+"+"+str(i)+")})"

print(out_idx_tensor)
print("\n")
print(out_value_tensor)
