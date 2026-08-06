; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @wrapped_slice(ptr noalias align 16 dereferenceable(117440512) %0, ptr noalias align 256 dereferenceable(14680064) %1) #0 {
  %3 = call i32 @llvm.amdgcn.workgroup.id.x(), !range !1
  %4 = call i32 @llvm.amdgcn.workitem.id.x(), !range !2
  %5 = mul i32 %4, 4
  %6 = mul i32 %3, 1024
  %7 = add i32 %5, %6
  %8 = getelementptr inbounds [58720256 x bfloat], ptr %0, i32 0, i32 %7
  %9 = load <4 x bfloat>, ptr %8, align 2, !invariant.load !3
  %10 = getelementptr inbounds [7340032 x bfloat], ptr %1, i32 0, i32 %7
  store <4 x bfloat> %9, ptr %10, align 2
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-max-num-workgroups"="7168,1,1" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 7168}
!2 = !{i32 0, i32 256}
!3 = !{}
