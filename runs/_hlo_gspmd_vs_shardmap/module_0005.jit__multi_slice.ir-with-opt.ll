; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @wrapped_slice_7(ptr noalias readonly align 16 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4194304) %1) local_unnamed_addr #0 {
  %wrapped_slice_7.kernarg.segment = call nonnull align 16 dereferenceable(16) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %.kernarg.offset5 = bitcast ptr addrspace(4) %wrapped_slice_7.kernarg.segment to ptr addrspace(4), !amdgpu.uniform !2
  %3 = load <2 x i64>, ptr addrspace(4) %.kernarg.offset5, align 16, !invariant.load !2
  %.load6 = extractelement <2 x i64> %3, i32 0
  %4 = inttoptr i64 %.load6 to ptr
  %.load47 = extractelement <2 x i64> %3, i32 1
  %5 = inttoptr i64 %.load47 to ptr
  %6 = addrspacecast ptr %5 to ptr addrspace(1)
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !range !3
  %9 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %10 = shl nuw nsw i32 %9, 2
  %11 = shl nuw nsw i32 %8, 10
  %12 = or disjoint i32 %10, %11
  %13 = shl nuw nsw i32 %9, 5
  %14 = and i32 %13, 4096
  %15 = shl nuw nsw i32 %8, 13
  %16 = or disjoint i32 %14, %15
  %17 = or disjoint i32 %16, %10
  %18 = or i32 %17, 3584
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %7, i64 %19
  %21 = load <4 x bfloat>, ptr addrspace(1) %20, align 2, !invariant.load !2, !alias.scope !5, !noalias !8, !amdgpu.noclobber !2
  %22 = zext nneg i32 %12 to i64
  %23 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %6, i64 %22
  store <4 x bfloat> %21, ptr addrspace(1) %23, align 2, !alias.scope !8, !noalias !5
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @wrapped_slice_6(ptr noalias readonly align 16 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4194304) %1) local_unnamed_addr #0 {
  %wrapped_slice_6.kernarg.segment = call nonnull align 16 dereferenceable(16) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %.kernarg.offset5 = bitcast ptr addrspace(4) %wrapped_slice_6.kernarg.segment to ptr addrspace(4), !amdgpu.uniform !2
  %3 = load <2 x i64>, ptr addrspace(4) %.kernarg.offset5, align 16, !invariant.load !2
  %.load6 = extractelement <2 x i64> %3, i32 0
  %4 = inttoptr i64 %.load6 to ptr
  %.load47 = extractelement <2 x i64> %3, i32 1
  %5 = inttoptr i64 %.load47 to ptr
  %6 = addrspacecast ptr %5 to ptr addrspace(1)
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !range !3
  %9 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %10 = shl nuw nsw i32 %9, 2
  %11 = shl nuw nsw i32 %8, 10
  %12 = or disjoint i32 %10, %11
  %13 = and i32 %10, 508
  %14 = shl nuw nsw i32 %9, 5
  %15 = and i32 %14, 4096
  %16 = or disjoint i32 %13, %15
  %17 = shl nuw nsw i32 %8, 13
  %18 = or disjoint i32 %16, %17
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %7, i64 %19
  %21 = getelementptr inbounds nuw i8, ptr addrspace(1) %20, i64 6144
  %22 = load <4 x bfloat>, ptr addrspace(1) %21, align 2, !invariant.load !2, !alias.scope !10, !noalias !13, !amdgpu.noclobber !2
  %23 = zext nneg i32 %12 to i64
  %24 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %6, i64 %23
  store <4 x bfloat> %22, ptr addrspace(1) %24, align 2, !alias.scope !13, !noalias !10
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @wrapped_slice_5(ptr noalias readonly align 16 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4194304) %1) local_unnamed_addr #0 {
  %wrapped_slice_5.kernarg.segment = call nonnull align 16 dereferenceable(16) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %.kernarg.offset5 = bitcast ptr addrspace(4) %wrapped_slice_5.kernarg.segment to ptr addrspace(4), !amdgpu.uniform !2
  %3 = load <2 x i64>, ptr addrspace(4) %.kernarg.offset5, align 16, !invariant.load !2
  %.load6 = extractelement <2 x i64> %3, i32 0
  %4 = inttoptr i64 %.load6 to ptr
  %.load47 = extractelement <2 x i64> %3, i32 1
  %5 = inttoptr i64 %.load47 to ptr
  %6 = addrspacecast ptr %5 to ptr addrspace(1)
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !range !3
  %9 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %10 = shl nuw nsw i32 %9, 2
  %11 = shl nuw nsw i32 %8, 10
  %12 = or disjoint i32 %10, %11
  %13 = shl nuw nsw i32 %9, 5
  %14 = and i32 %13, 4096
  %15 = shl nuw nsw i32 %8, 13
  %16 = or disjoint i32 %14, %15
  %17 = or disjoint i32 %16, %10
  %18 = or i32 %17, 2560
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %7, i64 %19
  %21 = load <4 x bfloat>, ptr addrspace(1) %20, align 2, !invariant.load !2, !alias.scope !15, !noalias !18, !amdgpu.noclobber !2
  %22 = zext nneg i32 %12 to i64
  %23 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %6, i64 %22
  store <4 x bfloat> %21, ptr addrspace(1) %23, align 2, !alias.scope !18, !noalias !15
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @wrapped_slice_4(ptr noalias readonly align 16 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4194304) %1) local_unnamed_addr #0 {
  %wrapped_slice_4.kernarg.segment = call nonnull align 16 dereferenceable(16) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %.kernarg.offset5 = bitcast ptr addrspace(4) %wrapped_slice_4.kernarg.segment to ptr addrspace(4), !amdgpu.uniform !2
  %3 = load <2 x i64>, ptr addrspace(4) %.kernarg.offset5, align 16, !invariant.load !2
  %.load6 = extractelement <2 x i64> %3, i32 0
  %4 = inttoptr i64 %.load6 to ptr
  %.load47 = extractelement <2 x i64> %3, i32 1
  %5 = inttoptr i64 %.load47 to ptr
  %6 = addrspacecast ptr %5 to ptr addrspace(1)
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !range !3
  %9 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %10 = shl nuw nsw i32 %9, 2
  %11 = shl nuw nsw i32 %8, 10
  %12 = or disjoint i32 %10, %11
  %13 = and i32 %10, 508
  %14 = shl nuw nsw i32 %9, 5
  %15 = and i32 %14, 4096
  %16 = or disjoint i32 %13, %15
  %17 = shl nuw nsw i32 %8, 13
  %18 = or disjoint i32 %16, %17
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %7, i64 %19
  %21 = getelementptr inbounds nuw i8, ptr addrspace(1) %20, i64 4096
  %22 = load <4 x bfloat>, ptr addrspace(1) %21, align 2, !invariant.load !2, !alias.scope !20, !noalias !23, !amdgpu.noclobber !2
  %23 = zext nneg i32 %12 to i64
  %24 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %6, i64 %23
  store <4 x bfloat> %22, ptr addrspace(1) %24, align 2, !alias.scope !23, !noalias !20
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @wrapped_slice_3(ptr noalias readonly align 16 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4194304) %1) local_unnamed_addr #0 {
  %wrapped_slice_3.kernarg.segment = call nonnull align 16 dereferenceable(16) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %.kernarg.offset5 = bitcast ptr addrspace(4) %wrapped_slice_3.kernarg.segment to ptr addrspace(4), !amdgpu.uniform !2
  %3 = load <2 x i64>, ptr addrspace(4) %.kernarg.offset5, align 16, !invariant.load !2
  %.load6 = extractelement <2 x i64> %3, i32 0
  %4 = inttoptr i64 %.load6 to ptr
  %.load47 = extractelement <2 x i64> %3, i32 1
  %5 = inttoptr i64 %.load47 to ptr
  %6 = addrspacecast ptr %5 to ptr addrspace(1)
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !range !3
  %9 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %10 = shl nuw nsw i32 %9, 2
  %11 = shl nuw nsw i32 %8, 10
  %12 = or disjoint i32 %10, %11
  %13 = shl nuw nsw i32 %9, 5
  %14 = and i32 %13, 4096
  %15 = shl nuw nsw i32 %8, 13
  %16 = or disjoint i32 %14, %15
  %17 = or disjoint i32 %16, %10
  %18 = or i32 %17, 1536
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %7, i64 %19
  %21 = load <4 x bfloat>, ptr addrspace(1) %20, align 2, !invariant.load !2, !alias.scope !25, !noalias !28, !amdgpu.noclobber !2
  %22 = zext nneg i32 %12 to i64
  %23 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %6, i64 %22
  store <4 x bfloat> %21, ptr addrspace(1) %23, align 2, !alias.scope !28, !noalias !25
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @wrapped_slice_2(ptr noalias readonly align 16 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4194304) %1) local_unnamed_addr #0 {
  %wrapped_slice_2.kernarg.segment = call nonnull align 16 dereferenceable(16) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %.kernarg.offset5 = bitcast ptr addrspace(4) %wrapped_slice_2.kernarg.segment to ptr addrspace(4), !amdgpu.uniform !2
  %3 = load <2 x i64>, ptr addrspace(4) %.kernarg.offset5, align 16, !invariant.load !2
  %.load6 = extractelement <2 x i64> %3, i32 0
  %4 = inttoptr i64 %.load6 to ptr
  %.load47 = extractelement <2 x i64> %3, i32 1
  %5 = inttoptr i64 %.load47 to ptr
  %6 = addrspacecast ptr %5 to ptr addrspace(1)
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !range !3
  %9 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %10 = shl nuw nsw i32 %9, 2
  %11 = shl nuw nsw i32 %8, 10
  %12 = or disjoint i32 %10, %11
  %13 = and i32 %10, 508
  %14 = shl nuw nsw i32 %9, 5
  %15 = and i32 %14, 4096
  %16 = or disjoint i32 %13, %15
  %17 = shl nuw nsw i32 %8, 13
  %18 = or disjoint i32 %16, %17
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %7, i64 %19
  %21 = getelementptr inbounds nuw i8, ptr addrspace(1) %20, i64 2048
  %22 = load <4 x bfloat>, ptr addrspace(1) %21, align 2, !invariant.load !2, !alias.scope !30, !noalias !33, !amdgpu.noclobber !2
  %23 = zext nneg i32 %12 to i64
  %24 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %6, i64 %23
  store <4 x bfloat> %22, ptr addrspace(1) %24, align 2, !alias.scope !33, !noalias !30
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @wrapped_slice_1(ptr noalias readonly align 16 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4194304) %1) local_unnamed_addr #0 {
  %wrapped_slice_1.kernarg.segment = call nonnull align 16 dereferenceable(16) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %.kernarg.offset5 = bitcast ptr addrspace(4) %wrapped_slice_1.kernarg.segment to ptr addrspace(4), !amdgpu.uniform !2
  %3 = load <2 x i64>, ptr addrspace(4) %.kernarg.offset5, align 16, !invariant.load !2
  %.load6 = extractelement <2 x i64> %3, i32 0
  %4 = inttoptr i64 %.load6 to ptr
  %.load47 = extractelement <2 x i64> %3, i32 1
  %5 = inttoptr i64 %.load47 to ptr
  %6 = addrspacecast ptr %5 to ptr addrspace(1)
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !range !3
  %9 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %10 = shl nuw nsw i32 %9, 2
  %11 = shl nuw nsw i32 %8, 10
  %12 = or disjoint i32 %10, %11
  %13 = shl nuw nsw i32 %9, 5
  %14 = and i32 %13, 4096
  %15 = shl nuw nsw i32 %8, 13
  %16 = or disjoint i32 %14, %15
  %17 = or disjoint i32 %16, %10
  %18 = or i32 %17, 512
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %7, i64 %19
  %21 = load <4 x bfloat>, ptr addrspace(1) %20, align 2, !invariant.load !2, !alias.scope !35, !noalias !38, !amdgpu.noclobber !2
  %22 = zext nneg i32 %12 to i64
  %23 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %6, i64 %22
  store <4 x bfloat> %21, ptr addrspace(1) %23, align 2, !alias.scope !38, !noalias !35
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @wrapped_slice(ptr noalias readonly align 16 captures(none) dereferenceable(33554432) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4194304) %1) local_unnamed_addr #0 {
  %wrapped_slice.kernarg.segment = call nonnull align 16 dereferenceable(16) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
  %.kernarg.offset5 = bitcast ptr addrspace(4) %wrapped_slice.kernarg.segment to ptr addrspace(4), !amdgpu.uniform !2
  %3 = load <2 x i64>, ptr addrspace(4) %.kernarg.offset5, align 16, !invariant.load !2
  %.load6 = extractelement <2 x i64> %3, i32 0
  %4 = inttoptr i64 %.load6 to ptr
  %.load47 = extractelement <2 x i64> %3, i32 1
  %5 = inttoptr i64 %.load47 to ptr
  %6 = addrspacecast ptr %5 to ptr addrspace(1)
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !range !3
  %9 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %10 = shl nuw nsw i32 %9, 2
  %11 = shl nuw nsw i32 %8, 10
  %12 = or disjoint i32 %10, %11
  %13 = and i32 %10, 508
  %14 = shl nuw nsw i32 %9, 5
  %15 = and i32 %14, 4096
  %16 = or disjoint i32 %13, %15
  %17 = shl nuw nsw i32 %8, 13
  %18 = or disjoint i32 %16, %17
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %7, i64 %19
  %21 = load <4 x bfloat>, ptr addrspace(1) %20, align 2, !invariant.load !2, !alias.scope !40, !noalias !43, !amdgpu.noclobber !2
  %22 = zext nneg i32 %12 to i64
  %23 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %6, i64 %22
  store <4 x bfloat> %21, ptr addrspace(1) %23, align 2, !alias.scope !43, !noalias !40
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr() #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="256,256" "amdgpu-max-num-workgroups"="2048,1,1" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!2 = !{}
!3 = !{i32 0, i32 2048}
!4 = !{i32 0, i32 256}
!5 = !{!6}
!6 = distinct !{!6, !7}
!7 = distinct !{!7, !"wrapped_slice_7"}
!8 = !{!9}
!9 = distinct !{!9, !7}
!10 = !{!11}
!11 = distinct !{!11, !12}
!12 = distinct !{!12, !"wrapped_slice_6"}
!13 = !{!14}
!14 = distinct !{!14, !12}
!15 = !{!16}
!16 = distinct !{!16, !17}
!17 = distinct !{!17, !"wrapped_slice_5"}
!18 = !{!19}
!19 = distinct !{!19, !17}
!20 = !{!21}
!21 = distinct !{!21, !22}
!22 = distinct !{!22, !"wrapped_slice_4"}
!23 = !{!24}
!24 = distinct !{!24, !22}
!25 = !{!26}
!26 = distinct !{!26, !27}
!27 = distinct !{!27, !"wrapped_slice_3"}
!28 = !{!29}
!29 = distinct !{!29, !27}
!30 = !{!31}
!31 = distinct !{!31, !32}
!32 = distinct !{!32, !"wrapped_slice_2"}
!33 = !{!34}
!34 = distinct !{!34, !32}
!35 = !{!36}
!36 = distinct !{!36, !37}
!37 = distinct !{!37, !"wrapped_slice_1"}
!38 = !{!39}
!39 = distinct !{!39, !37}
!40 = !{!41}
!41 = distinct !{!41, !42}
!42 = distinct !{!42, !"wrapped_slice"}
!43 = !{!44}
!44 = distinct !{!44, !42}
