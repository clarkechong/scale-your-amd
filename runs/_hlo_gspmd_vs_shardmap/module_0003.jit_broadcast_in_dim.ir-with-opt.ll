; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @wrapped_broadcast(ptr noalias readonly align 16 captures(none) dereferenceable(2) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(33554432) %1) local_unnamed_addr #0 {
  %wrapped_broadcast.kernarg.segment = call nonnull align 16 dereferenceable(16) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %.kernarg.offset5 = bitcast ptr addrspace(4) %wrapped_broadcast.kernarg.segment to ptr addrspace(4), !amdgpu.uniform !2
  %3 = load <2 x i64>, ptr addrspace(4) %.kernarg.offset5, align 16, !invariant.load !2
  %.load6 = extractelement <2 x i64> %3, i32 0
  %4 = inttoptr i64 %.load6 to ptr
  %.load47 = extractelement <2 x i64> %3, i32 1
  %5 = inttoptr i64 %.load47 to ptr
  %6 = addrspacecast ptr %5 to ptr addrspace(1)
  %7 = addrspacecast ptr %4 to ptr addrspace(1), !amdgpu.uniform !2
  %8 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !range !3
  %9 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %10 = load bfloat, ptr addrspace(1) %7, align 2, !invariant.load !2, !alias.scope !5, !noalias !8, !amdgpu.noclobber !2
  %11 = shl nuw nsw i32 %9, 2
  %12 = shl nuw nsw i32 %8, 10
  %13 = or disjoint i32 %11, %12
  %14 = insertelement <4 x bfloat> poison, bfloat %10, i64 0
  %15 = shufflevector <4 x bfloat> %14, <4 x bfloat> poison, <4 x i32> zeroinitializer
  %16 = zext nneg i32 %13 to i64
  %17 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %6, i64 %16
  store <4 x bfloat> %15, ptr addrspace(1) %17, align 2, !alias.scope !8, !noalias !5
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr() #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="256,256" "amdgpu-max-num-workgroups"="16384,1,1" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!2 = !{}
!3 = !{i32 0, i32 16384}
!4 = !{i32 0, i32 256}
!5 = !{!6}
!6 = distinct !{!6, !7}
!7 = distinct !{!7, !"wrapped_broadcast"}
!8 = !{!9}
!9 = distinct !{!9, !7}
