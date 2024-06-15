From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Inductive CtorTyp :=
  | CtorTyp_TBool
  | CtorTyp_TFun.

Inductive LeafCtorTyp :=
  | LeafCtorTyp_TBool.

Inductive CtorExpr :=
  | CtorExpr_Var
  | CtorExpr_Bool
  | CtorExpr_Abs
  | CtorExpr_App.

Inductive LeafCtorExpr :=
  | LeafCtorExpr_Var
  | LeafCtorExpr_Bool.

Inductive TupLeafCtorTypLeafCtorTyp :=
  | MkLeafCtorTypLeafCtorTyp : LeafCtorTyp -> LeafCtorTyp -> TupLeafCtorTypLeafCtorTyp.

Inductive TupCtorTypCtorTyp :=
  | MkCtorTypCtorTyp : CtorTyp -> CtorTyp -> TupCtorTypCtorTyp.

Inductive TupCtorExprCtorExpr :=
  | MkCtorExprCtorExpr : CtorExpr -> CtorExpr -> TupCtorExprCtorExpr.

Inductive TupLeafCtorExprLeafCtorExpr :=
  | MkLeafCtorExprLeafCtorExpr : LeafCtorExpr -> LeafCtorExpr -> TupLeafCtorExprLeafCtorExpr.

Inductive TupCtorTypLeafCtorExpr :=
  | MkCtorTypLeafCtorExpr : CtorTyp -> LeafCtorExpr -> TupCtorTypLeafCtorExpr.

Inductive TupCtorTypCtorExpr :=
  | MkCtorTypCtorExpr : CtorTyp -> CtorExpr -> TupCtorTypCtorExpr.

Definition genLeafTyp (chosen_ctor : LeafCtorTyp) (stack1 : nat) : G (Typ) :=
  match chosen_ctor with
  | LeafCtorTyp_TBool  => 
    (returnGen (TBool ))
  end.

Fixpoint genTyp (size : nat) (chosen_ctor : CtorTyp) (stack1 : nat) : G (Typ) :=
  match size with
  | O  => match chosen_ctor with
    | CtorTyp_TBool  => 
      (returnGen (TBool ))
    | CtorTyp_TFun  => 
      (bindGen 
      (* Frequency2 (single-branch) *) 
      (returnGen (MkLeafCtorTypLeafCtorTyp (LeafCtorTyp_TBool ) (LeafCtorTyp_TBool ))) 
      (fun param_variantis => (let '(MkLeafCtorTypLeafCtorTyp ctor1 ctor2) := param_variantis in

        (bindGen (genLeafTyp ctor1 1) 
        (fun p1 => 
          (bindGen (genLeafTyp ctor2 7) 
          (fun p2 => 
            (returnGen (TFun p1 p2)))))))))
    end
  | S size1 => match chosen_ctor with
    | CtorTyp_TBool  => 
      (returnGen (TBool ))
    | CtorTyp_TFun  => 
      (bindGen 
      (* Frequency3 *) (freq [
        (* 1 *) (match (size, stack1) with
        | (1, 3) => 92
        | (1, 5) => 96
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TBool ) (CtorTyp_TBool )))); 
        (* 2 *) (match (size, stack1) with
        | (1, 3) => 95
        | (1, 5) => 89
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TFun ) (CtorTyp_TBool )))); 
        (* 3 *) (match (size, stack1) with
        | (1, 3) => 89
        | (1, 5) => 92
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TBool ) (CtorTyp_TFun )))); 
        (* 4 *) (match (size, stack1) with
        | (1, 3) => 95
        | (1, 5) => 66
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorTyp (CtorTyp_TFun ) (CtorTyp_TFun ))))]) 
      (fun param_variantis => (let '(MkCtorTypCtorTyp ctor1 ctor2) := param_variantis in

        (bindGen (genTyp size1 ctor1 2) 
        (fun p1 => 
          (bindGen (genTyp size1 ctor2 8) 
          (fun p2 => 
            (returnGen (TFun p1 p2)))))))))
    end
  end.

Definition genLeafExpr (chosen_ctor : LeafCtorExpr) (stack1 : nat) : G (Expr) :=
  match chosen_ctor with
  | LeafCtorExpr_Var  => 
    (bindGen 
    (* GenNat1 *)
    (let _weight_1 := match (stack1) with
    | (10) => 10
    | (4) => 6
    | (9) => 5
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_1, returnGen 1);
      (100-_weight_1, returnGen 0)
    ]) (fun n1 =>
    (let _weight_2 := match (stack1) with
    | (10) => 91
    | (4) => 6
    | (9) => 7
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_2, returnGen 2);
      (100-_weight_2, returnGen 0)
    ]) (fun n2 =>
    (let _weight_4 := match (stack1) with
    | (10) => 93
    | (4) => 5
    | (9) => 93
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_4, returnGen 4);
      (100-_weight_4, returnGen 0)
    ]) (fun n4 =>
    (let _weight_8 := match (stack1) with
    | (10) => 91
    | (4) => 3
    | (9) => 7
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_8, returnGen 8);
      (100-_weight_8, returnGen 0)
    ]) (fun n8 =>
    (let _weight_16 := match (stack1) with
    | (10) => 7
    | (4) => 6
    | (9) => 94
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_16, returnGen 16);
      (100-_weight_16, returnGen 0)
    ]) (fun n16 =>
    (let _weight_32 := match (stack1) with
    | (10) => 90
    | (4) => 98
    | (9) => 93
    | _ => 500
    end
    in
    bindGen (freq [
      (_weight_32, returnGen 32);
      (100-_weight_32, returnGen 0)
    ]) (fun n32 =>
      returnGen (n1 + n2 + n4 + n8 + n16 + n32)
    )))))))))))) 
    (fun p1 => 
      (returnGen (Var p1))))
  | LeafCtorExpr_Bool  => 
    (bindGen 
    (* GenBool1 *) (let _weight_true := match (stack1) with
    | (10) => 42
    | (4) => 50
    | (9) => 4
    | _ => 500
    end
    in
    freq [
      (_weight_true, returnGen true);
      (100-_weight_true, returnGen false)
    ]) 
    (fun p1 => 
      (returnGen (Bool p1))))
  end.

Fixpoint genExpr (size : nat) (chosen_ctor : CtorExpr) (stack1 : nat) : G (Expr) :=
  match size with
  | O  => match chosen_ctor with
    | CtorExpr_Var  => 
      (bindGen 
      (* GenNat2 *)
      (let _weight_1 := match (stack1) with
      | (11) => 87
      | (12) => 73
      | (6) => 22
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (stack1) with
      | (11) => 58
      | (12) => 21
      | (6) => 79
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (stack1) with
      | (11) => 2
      | (12) => 23
      | (6) => 21
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
      (let _weight_8 := match (stack1) with
      | (11) => 4
      | (12) => 72
      | (6) => 13
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_8, returnGen 8);
        (100-_weight_8, returnGen 0)
      ]) (fun n8 =>
      (let _weight_16 := match (stack1) with
      | (11) => 46
      | (12) => 72
      | (6) => 86
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_16, returnGen 16);
        (100-_weight_16, returnGen 0)
      ]) (fun n16 =>
      (let _weight_32 := match (stack1) with
      | (11) => 22
      | (12) => 28
      | (6) => 18
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_32, returnGen 32);
        (100-_weight_32, returnGen 0)
      ]) (fun n32 =>
        returnGen (n1 + n2 + n4 + n8 + n16 + n32)
      )))))))))))) 
      (fun p1 => 
        (returnGen (Var p1))))
    | CtorExpr_Bool  => 
      (bindGen 
      (* GenBool2 *) (let _weight_true := match (stack1) with
      | (11) => 21
      | (12) => 75
      | (6) => 50
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1))))
    | CtorExpr_Abs  => 
      (bindGen 
      (* Frequency4 *) (freq [
        (* 1 *) (match (stack1) with
        | (11) => 50
        | (12) => 20
        | (6) => 1
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TBool ) (LeafCtorExpr_Var )))); 
        (* 2 *) (match (stack1) with
        | (11) => 88
        | (12) => 97
        | (6) => 97
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TFun ) (LeafCtorExpr_Var )))); 
        (* 3 *) (match (stack1) with
        | (11) => 48
        | (12) => 1
        | (6) => 4
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TBool ) (LeafCtorExpr_Bool )))); 
        (* 4 *) (match (stack1) with
        | (11) => 26
        | (12) => 42
        | (6) => 24
        | _ => 500
        end,
        (returnGen (MkCtorTypLeafCtorExpr (CtorTyp_TFun ) (LeafCtorExpr_Bool ))))]) 
      (fun param_variantis => (let '(MkCtorTypLeafCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genTyp 1 ctor1 3) 
        (fun p1 => 
          (bindGen (genLeafExpr ctor2 9) 
          (fun p2 => 
            (returnGen (Abs p1 p2)))))))))
    | CtorExpr_App  => 
      (bindGen 
      (* Frequency5 *) (freq [
        (* 1 *) (match (stack1) with
        | (11) => 95
        | (12) => 97
        | (6) => 97
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Var ) (LeafCtorExpr_Var )))); 
        (* 2 *) (match (stack1) with
        | (11) => 0
        | (12) => 0
        | (6) => 0
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Bool ) (LeafCtorExpr_Var )))); 
        (* 3 *) (match (stack1) with
        | (11) => 56
        | (12) => 38
        | (6) => 44
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Var ) (LeafCtorExpr_Bool )))); 
        (* 4 *) (match (stack1) with
        | (11) => 0
        | (12) => 0
        | (6) => 0
        | _ => 500
        end,
        (returnGen (MkLeafCtorExprLeafCtorExpr (LeafCtorExpr_Bool ) (LeafCtorExpr_Bool ))))]) 
      (fun param_variantis => (let '(MkLeafCtorExprLeafCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genLeafExpr ctor1 4) 
        (fun p1 => 
          (bindGen (genLeafExpr ctor2 10) 
          (fun p2 => 
            (returnGen (App p1 p2)))))))))
    end
  | S size1 => match chosen_ctor with
    | CtorExpr_Var  => 
      (bindGen 
      (* GenNat3 *)
      (let _weight_1 := match (size, stack1) with
      | (1, 11) => 55
      | (1, 12) => 5
      | (1, 6) => 9
      | (2, 11) => 43
      | (2, 12) => 61
      | (2, 6) => 76
      | (3, 11) => 41
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (size, stack1) with
      | (1, 11) => 70
      | (1, 12) => 47
      | (1, 6) => 65
      | (2, 11) => 52
      | (2, 12) => 25
      | (2, 6) => 55
      | (3, 11) => 60
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (size, stack1) with
      | (1, 11) => 29
      | (1, 12) => 11
      | (1, 6) => 87
      | (2, 11) => 41
      | (2, 12) => 16
      | (2, 6) => 66
      | (3, 11) => 48
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
      (let _weight_8 := match (size, stack1) with
      | (1, 11) => 20
      | (1, 12) => 8
      | (1, 6) => 94
      | (2, 11) => 41
      | (2, 12) => 69
      | (2, 6) => 34
      | (3, 11) => 54
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_8, returnGen 8);
        (100-_weight_8, returnGen 0)
      ]) (fun n8 =>
      (let _weight_16 := match (size, stack1) with
      | (1, 11) => 83
      | (1, 12) => 96
      | (1, 6) => 16
      | (2, 11) => 60
      | (2, 12) => 72
      | (2, 6) => 29
      | (3, 11) => 50
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_16, returnGen 16);
        (100-_weight_16, returnGen 0)
      ]) (fun n16 =>
      (let _weight_32 := match (size, stack1) with
      | (1, 11) => 46
      | (1, 12) => 49
      | (1, 6) => 23
      | (2, 11) => 59
      | (2, 12) => 41
      | (2, 6) => 91
      | (3, 11) => 52
      | (3, 12) => 50
      | (3, 6) => 50
      | (4, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_32, returnGen 32);
        (100-_weight_32, returnGen 0)
      ]) (fun n32 =>
        returnGen (n1 + n2 + n4 + n8 + n16 + n32)
      )))))))))))) 
      (fun p1 => 
        (returnGen (Var p1))))
    | CtorExpr_Bool  => 
      (bindGen 
      (* GenBool3 *) (let _weight_true := match (size, stack1) with
      | (1, 11) => 47
      | (1, 12) => 89
      | (1, 6) => 50
      | (2, 11) => 36
      | (2, 12) => 38
      | (2, 6) => 50
      | (3, 11) => 46
      | (3, 12) => 44
      | (3, 6) => 50
      | (4, 0) => 53
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1))))
    | CtorExpr_Abs  => 
      (bindGen 
      (* Frequency6 *) (freq [
        (* 1 *) (match (size, stack1) with
        | (1, 11) => 35
        | (1, 12) => 32
        | (1, 6) => 21
        | (2, 11) => 40
        | (2, 12) => 2
        | (2, 6) => 10
        | (3, 11) => 17
        | (3, 12) => 50
        | (3, 6) => 48
        | (4, 0) => 0
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_Var )))); 
        (* 2 *) (match (size, stack1) with
        | (1, 11) => 38
        | (1, 12) => 83
        | (1, 6) => 84
        | (2, 11) => 59
        | (2, 12) => 3
        | (2, 6) => 10
        | (3, 11) => 33
        | (3, 12) => 51
        | (3, 6) => 52
        | (4, 0) => 0
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_Var )))); 
        (* 3 *) (match (size, stack1) with
        | (1, 11) => 6
        | (1, 12) => 4
        | (1, 6) => 0
        | (2, 11) => 28
        | (2, 12) => 2
        | (2, 6) => 5
        | (3, 11) => 8
        | (3, 12) => 50
        | (3, 6) => 52
        | (4, 0) => 0
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_Bool )))); 
        (* 4 *) (match (size, stack1) with
        | (1, 11) => 51
        | (1, 12) => 1
        | (1, 6) => 2
        | (2, 11) => 44
        | (2, 12) => 3
        | (2, 6) => 7
        | (3, 11) => 11
        | (3, 12) => 50
        | (3, 6) => 47
        | (4, 0) => 0
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_Bool )))); 
        (* 5 *) (match (size, stack1) with
        | (1, 11) => 55
        | (1, 12) => 5
        | (1, 6) => 71
        | (2, 11) => 60
        | (2, 12) => 4
        | (2, 6) => 42
        | (3, 11) => 42
        | (3, 12) => 50
        | (3, 6) => 49
        | (4, 0) => 0
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_Abs )))); 
        (* 6 *) (match (size, stack1) with
        | (1, 11) => 80
        | (1, 12) => 94
        | (1, 6) => 92
        | (2, 11) => 60
        | (2, 12) => 69
        | (2, 6) => 44
        | (3, 11) => 60
        | (3, 12) => 53
        | (3, 6) => 52
        | (4, 0) => 0
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_Abs )))); 
        (* 7 *) (match (size, stack1) with
        | (1, 11) => 31
        | (1, 12) => 12
        | (1, 6) => 10
        | (2, 11) => 45
        | (2, 12) => 79
        | (2, 6) => 77
        | (3, 11) => 77
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 11
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TBool ) (CtorExpr_App )))); 
        (* 8 *) (match (size, stack1) with
        | (1, 11) => 79
        | (1, 12) => 80
        | (1, 6) => 90
        | (2, 11) => 58
        | (2, 12) => 94
        | (2, 6) => 92
        | (3, 11) => 85
        | (3, 12) => 46
        | (3, 6) => 50
        | (4, 0) => 99
        | _ => 500
        end,
        (returnGen (MkCtorTypCtorExpr (CtorTyp_TFun ) (CtorExpr_App ))))]) 
      (fun param_variantis => (let '(MkCtorTypCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genTyp 1 ctor1 5) 
        (fun p1 => 
          (bindGen (genExpr size1 ctor2 11) 
          (fun p2 => 
            (returnGen (Abs p1 p2)))))))))
    | CtorExpr_App  => 
      (bindGen 
      (* Frequency7 *) (freq [
        (* 1 *) (match (size, stack1) with
        | (1, 11) => 79
        | (1, 12) => 70
        | (1, 6) => 15
        | (2, 11) => 54
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 46
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_Var )))); 
        (* 2 *) (match (size, stack1) with
        | (1, 11) => 1
        | (1, 12) => 0
        | (1, 6) => 0
        | (2, 11) => 18
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 46
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_Var )))); 
        (* 3 *) (match (size, stack1) with
        | (1, 11) => 59
        | (1, 12) => 96
        | (1, 6) => 84
        | (2, 11) => 64
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 46
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_Var )))); 
        (* 4 *) (match (size, stack1) with
        | (1, 11) => 80
        | (1, 12) => 14
        | (1, 6) => 23
        | (2, 11) => 58
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 46
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_Var )))); 
        (* 5 *) (match (size, stack1) with
        | (1, 11) => 9
        | (1, 12) => 2
        | (1, 6) => 1
        | (2, 11) => 42
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 46
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_Bool )))); 
        (* 6 *) (match (size, stack1) with
        | (1, 11) => 1
        | (1, 12) => 0
        | (1, 6) => 0
        | (2, 11) => 18
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 46
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_Bool )))); 
        (* 7 *) (match (size, stack1) with
        | (1, 11) => 27
        | (1, 12) => 4
        | (1, 6) => 1
        | (2, 11) => 53
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 53
        | (4, 0) => 66
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_Bool )))); 
        (* 8 *) (match (size, stack1) with
        | (1, 11) => 42
        | (1, 12) => 16
        | (1, 6) => 0
        | (2, 11) => 55
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 51
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_Bool )))); 
        (* 9 *) (match (size, stack1) with
        | (1, 11) => 34
        | (1, 12) => 92
        | (1, 6) => 96
        | (2, 11) => 56
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 46
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_Abs )))); 
        (* 10 *) (match (size, stack1) with
        | (1, 11) => 1
        | (1, 12) => 0
        | (1, 6) => 0
        | (2, 11) => 18
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 46
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_Abs )))); 
        (* 11 *) (match (size, stack1) with
        | (1, 11) => 71
        | (1, 12) => 96
        | (1, 6) => 98
        | (2, 11) => 69
        | (2, 12) => 1
        | (2, 6) => 1
        | (3, 11) => 0
        | (3, 12) => 54
        | (3, 6) => 52
        | (4, 0) => 68
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_Abs )))); 
        (* 12 *) (match (size, stack1) with
        | (1, 11) => 84
        | (1, 12) => 90
        | (1, 6) => 97
        | (2, 11) => 68
        | (2, 12) => 4
        | (2, 6) => 7
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 48
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_Abs )))); 
        (* 13 *) (match (size, stack1) with
        | (1, 11) => 84
        | (1, 12) => 81
        | (1, 6) => 28
        | (2, 11) => 54
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 46
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Var ) (CtorExpr_App )))); 
        (* 14 *) (match (size, stack1) with
        | (1, 11) => 1
        | (1, 12) => 0
        | (1, 6) => 0
        | (2, 11) => 18
        | (2, 12) => 0
        | (2, 6) => 0
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 50
        | (4, 0) => 46
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Bool ) (CtorExpr_App )))); 
        (* 15 *) (match (size, stack1) with
        | (1, 11) => 85
        | (1, 12) => 95
        | (1, 6) => 94
        | (2, 11) => 65
        | (2, 12) => 1
        | (2, 6) => 5
        | (3, 11) => 0
        | (3, 12) => 50
        | (3, 6) => 51
        | (4, 0) => 53
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_Abs ) (CtorExpr_App )))); 
        (* 16 *) (match (size, stack1) with
        | (1, 11) => 77
        | (1, 12) => 95
        | (1, 6) => 80
        | (2, 11) => 59
        | (2, 12) => 99
        | (2, 6) => 99
        | (3, 11) => 99
        | (3, 12) => 51
        | (3, 6) => 50
        | (4, 0) => 46
        | _ => 500
        end,
        (returnGen (MkCtorExprCtorExpr (CtorExpr_App ) (CtorExpr_App ))))]) 
      (fun param_variantis => (let '(MkCtorExprCtorExpr ctor1 ctor2) := param_variantis in

        (bindGen (genExpr size1 ctor1 6) 
        (fun p1 => 
          (bindGen (genExpr size1 ctor2 12) 
          (fun p2 => 
            (returnGen (App p1 p2)))))))))
    end
  end.

Definition gSized :=

  (bindGen 
  (* Frequency1 *) (freq [
    (* 1 *) (match (tt) with
    | tt => 0
    end,
    (returnGen (CtorExpr_Var ))); 
    (* 2 *) (match (tt) with
    | tt => 0
    end,
    (returnGen (CtorExpr_Bool ))); 
    (* 3 *) (match (tt) with
    | tt => 97
    end,
    (returnGen (CtorExpr_Abs ))); 
    (* 4 *) (match (tt) with
    | tt => 0
    end,
    (returnGen (CtorExpr_App )))]) 
  (fun init_ctor => (genExpr 4 init_ctor 0))).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
