; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @wrapped_broadcast(ptr noalias align 16 dereferenceable(2) %0, ptr noalias align 256 dereferenceable(117440512) %1) #0 {
  %3 = call i32 @llvm.amdgcn.workgroup.id.x(), !range !1
  %4 = call i32 @llvm.amdgcn.workitem.id.x(), !range !2
  %5 = getelementptr inbounds [1 x bfloat], ptr %0, i32 0, i32 0
  %6 = load bfloat, ptr %5, align 2, !invariant.load !3
  %7 = mul i32 %4, 4
  %8 = mul i32 %3, 1024
  %9 = add i32 %7, %8
  %10 = insertelement <4 x bfloat> poison, bfloat %6, i32 0
  %11 = shufflevector <4 x bfloat> %10, <4 x bfloat> poison, <4 x i32> zeroinitializer
  %12 = getelementptr inbounds [58720256 x bfloat], ptr %1, i32 0, i32 %9
  store <4 x bfloat> %11, ptr %12, align 2
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-max-num-workgroups"="57344,1,1" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 57344}
!2 = !{i32 0, i32 256}
!3 = !{}
